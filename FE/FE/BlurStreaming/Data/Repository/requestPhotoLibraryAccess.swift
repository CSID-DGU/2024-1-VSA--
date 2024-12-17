//
//  requestPhotoLibraryAccess.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/17/24.
//

import Foundation
import Photos

public func requestPhotoLibraryAccess(completion: @escaping (Bool) -> Void) {
    let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)

    switch status {
    case .authorized:
        completion(true)
    case .notDetermined:
        PHPhotoLibrary.requestAuthorization(for: .readWrite) { newStatus in
            DispatchQueue.main.async {
                completion(newStatus == .authorized)
            }
        }
    default:
        completion(false)
    }
}
