import SwiftUI

struct ResultView: View {
    let result: ImageResponse
    
    var body: some View {
        VStack(spacing: 20) {
            
            Text("딥페이크 위험도 \(Int(result.riskScore * 100))%")
                .font(.title)
                .bold()
            
            AsyncImage(url: URL(string: result.processedUrl)) { image in
                image.resizable()
            } placeholder: {
                ProgressView()
            }
            .frame(height: 250)
            
            HStack {
                Button("공유") {
                    print("공유")
                }
                
                Button("저장") {
                    print("저장")
                }
            }
        }
        .padding()
    }
}
