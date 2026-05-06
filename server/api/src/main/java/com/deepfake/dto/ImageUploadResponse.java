package com.deepfake.dto;

public class ImageUploadResponse {
    private Long imageId;
    private String resultUrl;
    private AnalyzeResult analyzeResult;

    public ImageUploadResponse(Long imageId, String resultUrl, AnalyzeResult analyzeResult) {
        this.imageId = imageId;
        this.resultUrl = resultUrl;
        this.analyzeResult = analyzeResult;
    }

    public Long getImageId() { return imageId; }
    public String getResultUrl() { return resultUrl; }
    public AnalyzeResult getAnalyzeResult() { return analyzeResult; }
}