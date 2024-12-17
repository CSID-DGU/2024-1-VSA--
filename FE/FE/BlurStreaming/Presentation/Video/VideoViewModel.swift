//
//  VideoViewModel.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/17/24.
//

import Foundation
import RxSwift

final class VideoViewModel {
    
    private let network: VideoNetwork
    
    init() {
        let networkProvider = NetworkProvider()
        self.network = networkProvider.makeVideoNetwork()
    }
    
    struct Input {
        let updateTrigger: PublishSubject<Void>
    }
    
    struct Output {
        let video:Observable<VideoResponse>
    }
    
    public func transform(input:Input) -> Output {
        let videos = input.updateTrigger.flatMapLatest {_ -> Observable<VideoResponse> in
            return self.network.getVideoList()
        }
        
        return Output(video: videos)
    }
}
