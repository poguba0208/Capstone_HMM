import SwiftUI

struct RiskResultView: View {
    let image: UIImage
    let analyzeResult: AnalyzeResponse
    let onApplyFilter: () -> Void
    let onDismiss: () -> Void

    var scorePercent: Double { analyzeResult.risk.score }

    var gaugeColor: Color {
        if scorePercent >= 0.7 { return .red }
        if scorePercent >= 0.4 { return .orange }
        return .green
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {

                // 원본 이미지
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .cornerRadius(14)
                    .padding(.horizontal)
                    .padding(.top, 16)

                // 위험도 카드
                VStack(spacing: 14) {
                    Text("딥페이크 위험도")
                        .font(.headline)

                    // 점수 숫자
                    Text(String(format: "%.0f%%", scorePercent * 100))
                        .font(.system(size: 48, weight: .bold, design: .rounded))
                        .foregroundColor(gaugeColor)

                    // 게이지 바
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color(.systemGray5))
                                .frame(height: 14)
                            RoundedRectangle(cornerRadius: 8)
                                .fill(gaugeColor)
                                .frame(width: geo.size.width * scorePercent, height: 14)
                                .animation(.easeOut(duration: 0.6), value: scorePercent)
                        }
                    }
                    .frame(height: 14)

                    // 범례
                    HStack {
                        Text("낮음").font(.caption).foregroundColor(.green)
                        Spacer()
                        Text("중간").font(.caption).foregroundColor(.orange)
                        Spacer()
                        Text("높음").font(.caption).foregroundColor(.red)
                    }

                    if analyzeResult.faceCount == 0 {
                        Text("얼굴이 감지되지 않았습니다")
                            .font(.subheadline)
                            .foregroundColor(.gray)
                            .padding(.top, 4)
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(16)
                .padding(.horizontal)

                // 필터 적용 버튼 (항상 표시)
                Button(action: onApplyFilter) {
                    HStack {
                        Image(systemName: "sparkles")
                        Text("딥페이크 방지 필터 적용")
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(
                        LinearGradient(colors: [.blue, .purple],
                                       startPoint: .leading,
                                       endPoint: .trailing)
                    )
                    .cornerRadius(12)
                }
                .padding(.horizontal)

                Button("취소") { onDismiss() }
                    .foregroundColor(.gray)
                    .font(.subheadline)
                    .padding(.bottom, 20)
            }
        }
    }
}

#Preview {
    RiskResultView(
        image: UIImage(systemName: "person.fill")!,
        analyzeResult: AnalyzeResponse(
            faceCount: 1,
            faces: [
                AnalyzeResponse.FaceDetail(
                    faceRatio: 0.45,
                    headPose: AnalyzeResponse.HeadPose(yaw: 12.3, pitch: -5.1)
                )
            ],
            risk: AnalyzeResponse.RiskInfo(score: 0.72, level: "HIGH")
        ),
        onApplyFilter: {},
        onDismiss: {}
    )
}
