//
//  NetworkProvider.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/17/24.
//

import Foundation
final class NetworkProvider {
    private let endpoint: String
    private let token: String
    
    init() {
        self.endpoint = "http://127.0.0.1:8000/testapp"
        self.token = "token"
    }
   
    
    public func makeVideoNetwork() -> VideoNetwork {
        let videoNetwork = Network<VideoResponse>(self.endpoint, token: self.token)
        
        return VideoNetwork(videoNetwork: videoNetwork)
    }
}
