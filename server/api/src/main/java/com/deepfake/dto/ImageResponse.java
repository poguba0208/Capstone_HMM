package com.deepfake.dto;

public class ImageResponse {

    private Long id;

    private String fileName;

    private String url;

    private String status;

    private Double riskScore;

    private String resultPath;

    private String errorMessage;

    public ImageResponse(
            Long id,
            String fileName,
            String url,
            String status,
            Double riskScore,
            String resultPath,
            String errorMessage
    ) {
        this.id = id;
        this.fileName = fileName;
        this.url = url;
        this.status = status;
        this.riskScore = riskScore;
        this.resultPath = resultPath;
        this.errorMessage = errorMessage;
    }

    public Long getId() {
        return id;
    }

    public String getFileName() {
        return fileName;
    }

    public String getUrl() {
        return url;
    }

    public String getStatus() {
        return status;
    }

    public Double getRiskScore() {
        return riskScore;
    }

    public String getResultPath() {
        return resultPath;
    }

    public String getErrorMessage() {
        return errorMessage;
    }
}