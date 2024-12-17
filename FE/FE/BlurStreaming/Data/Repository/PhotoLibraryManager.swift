//
//  PhotoLibraryManager.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/16/24.
//

import Foundation
import Photos
import RxSwift

class PhotoLibraryManager {
    static let shared = PhotoLibraryManager()

    func saveVideoToPhotoLibrary(videoURL: URL) -> Observable<Void> {

        return Observable.create { observer in
            PHPhotoLibrary.requestAuthorization { status in
                guard status == .authorized else {
                    observer.onError(NSError(domain: "PhotoLibraryError", code: -1, userInfo: [NSLocalizedDescriptionKey: "사진 접근 권한이 없습니다."]))
                    return
                }

                PHPhotoLibrary.shared().performChanges({
                    PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: videoURL)
                }) { success, error in
                    if let error = error {
                        observer.onError(error)
                    } else if success {
                        observer.onNext(())
                        observer.onCompleted()
                    }
                }
            }
            return Disposables.create()
        }
    }
    
    /// 사진 접근 권한 확인 및 요청
    func requestPhotoLibraryAccess(completion: @escaping (Bool) -> Void) {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)

        switch status {
        case .authorized, .limited:
            // 이미 접근 권한이 허용됨
            completion(true)

        case .notDetermined:
            // 권한 요청
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { newStatus in
                DispatchQueue.main.async {
                    completion(newStatus == .authorized || newStatus == .limited)
                }
            }

        case .denied, .restricted:
            // 접근이 거부됨
            completion(false)

        @unknown default:
            // 알 수 없는 상태
            completion(false)
        }
    }
}
