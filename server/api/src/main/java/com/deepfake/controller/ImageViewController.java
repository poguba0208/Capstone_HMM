package com.deepfake.controller;

import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.File;
import java.nio.file.Files;

@RestController
@RequestMapping("/view")
public class ImageViewController {

    @GetMapping("/{filename}")
    public ResponseEntity<Resource> viewImage(
            @PathVariable String filename
    ) throws Exception {

        String path =
                System.getProperty("user.dir")
                        + "/uploads/"
                        + filename;

        File file = new File(path);

        Resource resource =
                new UrlResource(file.toURI());

        String contentType =
                Files.probeContentType(file.toPath());

        if (contentType == null) {
            contentType = "application/octet-stream";
        }

        return ResponseEntity.ok()
                .contentType(
                        MediaType.parseMediaType(contentType)
                )
                .header(
                        HttpHeaders.CONTENT_DISPOSITION,
                        "inline; filename=\"" + filename + "\""
                )
                .body(resource);
    }
}