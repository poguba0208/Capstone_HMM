package com.deepfake.external;

import com.deepfake.dto.AnalyzeResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Component
public class FaceShieldClient {

    @Value("${ai.server.url}")
    private String aiServerUrl;

    private final RestTemplate restTemplate;

    public FaceShieldClient(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public AnalyzeResult analyze(MultipartFile file) throws IOException {

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        ByteArrayResource fileResource =
                new ByteArrayResource(file.getBytes()) {

                    @Override
                    public String getFilename() {
                        return file.getOriginalFilename();
                    }
                };

        MultiValueMap<String, Object> body =
                new LinkedMultiValueMap<>();

        body.add("file", fileResource);

        HttpEntity<MultiValueMap<String, Object>> request =
                new HttpEntity<>(body, headers);

        ResponseEntity<AnalyzeResult> response =
                restTemplate.postForEntity(
                        aiServerUrl + "/analyze",
                        request,
                        AnalyzeResult.class
                );

        if (response.getBody() == null) {
            throw new RuntimeException("AI 분석 실패");
        }

        return response.getBody();
    }
}