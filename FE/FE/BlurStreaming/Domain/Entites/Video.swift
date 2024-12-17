//
//  Video.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/17/24.
//

import Foundation

struct VideoResponse: Decodable {
    let code: String
    let result: String
    let message: String
    let data: Video
    
    init() {
        self.code = "401"
        self.result = "FAILURE"
        self.message = "서버 연결에 실패했습니다."
        self.data = Video()
    }
}

struct Video: Decodable, Hashable {
    let videos: [String]
    
    private enum CodingKeys: String, CodingKey {
        case videos
    }
    
    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.videos = try container.decode([String].self, forKey: .videos)
    }
    
    init() {
        self.videos = []
    }
}
