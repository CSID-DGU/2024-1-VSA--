//
//  VideoNetwork.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/17/24.
//

import Foundation
import RxSwift

final class VideoNetwork {
    private let videoNetwork: Network<VideoResponse>
    
    init(videoNetwork: Network<VideoResponse>) {
        self.videoNetwork = videoNetwork
    }
    
    public func getVideoList() -> Observable<VideoResponse> {
        return self.videoNetwork.getItemList(path: "api/videos", defaultValue: VideoResponse())
    }
}
