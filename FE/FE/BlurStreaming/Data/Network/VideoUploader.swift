//
//  VideoUploader.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/16/24.
//

import Foundation
import RxSwift
import RxAlamofire
import Alamofire

class VideoUploader {
    func uploadVideo(videoURL: URL, to serverURL: URL) -> Observable<Double> {
        return Observable.create { observer in
            // Boundary 생성
            let boundary = "Boundary-\(UUID().uuidString)"
            var urlRequest = URLRequest(url: serverURL)
            urlRequest.httpMethod = "POST"
            urlRequest.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

            // 업로드 데이터 준비
            var body = Data()
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(videoURL.lastPathComponent)\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: video/mp4\r\n\r\n".data(using: .utf8)!)
            
            if let videoData = try? Data(contentsOf: videoURL) {
                body.append(videoData)
            }
            body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

            // Alamofire 요청 생성
            let uploadRequest = AF.upload(multipartFormData: { formData in
                formData.append(videoURL, withName: "file", fileName: videoURL.lastPathComponent, mimeType: "video/mp4")
            }, to: serverURL)

            // Rx 업로드 처리
            uploadRequest
                .uploadProgress { progress in
                    let progressValue = progress.fractionCompleted
                    observer.onNext(progressValue) // 진행 상황 전송
                }
                .response { response in
                    if let error = response.error {
                        observer.onError(error) // 에러 전송
                    } else {
                        observer.onCompleted() // 완료 전송
                    }
                }

            return Disposables.create {
                uploadRequest.cancel() // 업로드 취소 처리
            }
        }
    }
}
