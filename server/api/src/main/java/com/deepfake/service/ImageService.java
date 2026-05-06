package com.deepfake.service;
import com.deepfake.domain.Image;
import com.deepfake.domain.User;
import com.deepfake.dto.AnalyzeResult;
import com.deepfake.dto.ImageResponse;
import com.deepfake.dto.ImageUploadResponse;
import com.deepfake.external.FaceShieldClient;
import com.deepfake.repository.ImageRepository;
import com.deepfake.repository.UserRepository;
import com.deepfake.entity.ImageStatus;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class ImageService {

    private final ImageRepository imageRepository;
    private final UserRepository userRepository;
    private final FaceShieldClient faceShieldClient;

    @Autowired
    public ImageService(
            ImageRepository imageRepository,
            UserRepository userRepository,
            FaceShieldClient faceShieldClient
    ) {
        this.imageRepository = imageRepository;
        this.userRepository = userRepository;
        this.faceShieldClient = faceShieldClient;
    }

    public ImageUploadResponse uploadImage(
            MultipartFile file,
            Long userId
    ) {

        try {

            User user = null;

            if (userId != null) {

                user = userRepository.findById(userId)
                        .orElseThrow(() ->
                                new RuntimeException("유저 없음"));
            }

            String savedName = saveFile(file);

            Image image = new Image(
                    file.getOriginalFilename(),
                    savedName,
                    user
            );

            image.setStatus(ImageStatus.PENDING);

            imageRepository.save(image);

            analyzeAsync(image.getId(), file);

            String url =
                    "http://localhost:8080/uploads/"
                            + savedName;

            return new ImageUploadResponse(
                    image.getId(),
                    url,
                    null
            );

        } catch (IOException e) {

            throw new RuntimeException("파일 저장 실패");
        }
    }

    @Async
    public void analyzeAsync(
            Long imageId,
            MultipartFile file
    ) {

        Image image =
                imageRepository.findById(imageId)
                        .orElseThrow(() ->
                                new RuntimeException("이미지 없음"));

        try {

            image.setStatus(ImageStatus.ANALYZING);

            imageRepository.save(image);

            AnalyzeResult result =
                    faceShieldClient.analyze(file);

            if (result.getRisk() != null) {

                image.setRiskScore(
                        result.getRisk().getScore()
                );
            }

            image.setStatus(ImageStatus.COMPLETED);

            imageRepository.save(image);

        } catch (Exception e) {

            image.setStatus(ImageStatus.FAILED);

            image.setErrorMessage(e.getMessage());

            imageRepository.save(image);
        }
    }

    public List<ImageResponse> getMyImages(Long userId) {

        User user = userRepository.findById(userId)
                .orElseThrow(() ->
                        new RuntimeException("유저 없음"));

        return imageRepository.findByUser(user)
                .stream()
                .map(img -> new ImageResponse(
                        img.getId(),
                        img.getFileName(),
                        "http://localhost:8080/uploads/"
                                + img.getFilePath(),
                        img.getStatus().name(),
                        img.getRiskScore(),
                        img.getResultPath(),
                        img.getErrorMessage()
                ))
                .collect(Collectors.toList());
    }

    public ImageResponse getImageResult(Long imageId) {

        Image img = imageRepository.findById(imageId)
                .orElseThrow(() ->
                        new RuntimeException("이미지 없음"));

        return new ImageResponse(
                img.getId(),
                img.getFileName(),
                "http://localhost:8080/uploads/"
                        + img.getFilePath(),
                img.getStatus().name(),
                img.getRiskScore(),
                img.getResultPath(),
                img.getErrorMessage()
        );
    }

    private String saveFile(MultipartFile file)
            throws IOException {

        String uploadDir =
                System.getProperty("user.dir")
                        + "/uploads/";

        File dir = new File(uploadDir);

        if (!dir.exists()) {
            dir.mkdirs();
        }

        String originalName =
                file.getOriginalFilename();

        String extension =
                originalName.substring(
                        originalName.lastIndexOf(".")
                );

        String savedName =
                UUID.randomUUID() + extension;

        File saveFile =
                new File(uploadDir + savedName);

        file.transferTo(saveFile);

        return saveFile.getName();
    }
}