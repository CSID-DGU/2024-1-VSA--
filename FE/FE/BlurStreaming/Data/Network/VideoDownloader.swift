//
//  VideoDownloader.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/16/24.
//

import Foundation
import RxAlamofire
import RxSwift
import Alamofire

class VideoDownloader {
    static let shared = VideoDownloader()

    func downloadVideo(from url: URL, to destinationURL: URL) -> Observable<URL> {
        // 중복 파일 처리
        let uniqueDestinationURL = self.makeUniqueFileURL(originalURL: destinationURL)

        return RxAlamofire.download(URLRequest(url: url), to: { _, _ in
            (uniqueDestinationURL, .createIntermediateDirectories)
        })
        .flatMap { request -> Observable<URL> in
            return Observable.create { observer in
                request.response { response in
                    if let error = response.error {
                        observer.onError(error)
                    } else if let fileURL = response.fileURL {
                        observer.onNext(fileURL)
                        observer.onCompleted()
                    } else {
                        observer.onError(NSError(domain: "DownloadError", code: -1, userInfo: [NSLocalizedDescriptionKey: "다운로드된 파일 경로를 찾을 수 없습니다."]))
                    }
                }
                return Disposables.create {
                    request.cancel()
                }
            }
        }
    }

    // 파일 이름 중복 처리
    private func makeUniqueFileURL(originalURL: URL) -> URL {
        let fileManager = FileManager.default
        var uniqueURL = originalURL
        var counter = 1

        while fileManager.fileExists(atPath: uniqueURL.path) {
            let fileName = originalURL.deletingPathExtension().lastPathComponent
            let fileExtension = originalURL.pathExtension
            uniqueURL = originalURL.deletingLastPathComponent()
                .appendingPathComponent("\(fileName)-\(counter).\(fileExtension)")
            counter += 1
        }

        return uniqueURL
    }
}
