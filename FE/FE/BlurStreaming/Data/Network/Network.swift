//
//  Network.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/17/24.
//

import Foundation
import RxSwift
import RxAlamofire
import Alamofire

class Network<T:Decodable> {
    
    private let endpoint: String
    private let queue: ConcurrentDispatchQueueScheduler
    private let token: String
    
    init(_ endpoint:String, token:String) {
        self.endpoint = endpoint
        self.token = token
        self.queue = ConcurrentDispatchQueueScheduler(qos: .background)
    }
    
    func getItemList(path:String, defaultValue: T) -> Observable<T> {
        print("networktoken", self.token)
        let fullPath = "\(endpoint)/\(path)"
        print(fullPath)
        return RxAlamofire.data(.get, fullPath, headers:["accept": "*/*", "Authorization": "Bearer \(self.token)"])
            .map { data -> T in
                return try JSONDecoder().decode(T.self, from:data)
            }
            .catchError { error in
                    print("Error occurred: \(error)")
                    return Observable.just(defaultValue) // 기본값 반환
            }
            .observeOn(self.queue)
            .debug()
    }
    
    func PostData(path:String, defaultValue: T, updateData: [String: Any]) -> Observable<T> {
        let fullPath = "\(endpoint)/\(path)"
        print(fullPath)
        return RxAlamofire.data(.post, fullPath, parameters: updateData, encoding: JSONEncoding.default, headers:["accept": "*/*", "Authorization": "Bearer \(self.token)"])
            .map { data -> T in
                return try JSONDecoder().decode(T.self, from:data)
            }
            .catchError { error in
                    print("Error occurred: \(error)")
                    return Observable.just(defaultValue) // 기본값 반환
            }
            .observeOn(self.queue)
            .debug()
    }
    
    func PostDataWithoutToken(path:String, defaultValue: T, updateData: [String: Any]) -> Observable<T> {
        let fullPath = "\(endpoint)/\(path)"
        print(fullPath)
        return RxAlamofire.data(.post, fullPath, parameters: updateData, encoding: JSONEncoding.default, headers:["accept": "*/*"])
            .map { data -> T in
                return try JSONDecoder().decode(T.self, from:data)
            }
            .catchError { error in
                    print("Error occurred: \(error)")
                    return Observable.just(defaultValue) // 기본값 반환
            }
            .observeOn(self.queue)
            .debug()
    }
    
    func PatchData(path:String, defaultValue: T, updateData: [String: Any]) -> Observable<T> {
        let fullPath = "\(endpoint)/\(path)"
        print(fullPath)
        return RxAlamofire.data(.patch, fullPath, parameters: updateData, encoding: JSONEncoding.default, headers:["accept": "*/*", "Authorization": "Bearer \(self.token)"])
            .map { data -> T in
                return try JSONDecoder().decode(T.self, from:data)
            }
            .catchError { error in
                    print("Error occurred: \(error)")
                    return Observable.just(defaultValue) // 기본값 반환
            }
            .observeOn(self.queue)
            .debug()
    }
    
    func deleteData(path:String, defaultValue: T) -> Observable<T> {
        let fullPath = "\(endpoint)/\(path)"
        print(fullPath)
        return RxAlamofire.data(.delete, fullPath, headers:["accept": "*/*", "Authorization": "Bearer \(self.token)"])
            .map { data -> T in
                return try JSONDecoder().decode(T.self, from:data)
            }
            .catchError { error in
                    print("Error occurred: \(error)")
                    return Observable.just(defaultValue) // 기본값 반환
            }
            .observeOn(self.queue)
            .debug()
    }
    
    func KakaoPostData(parameters: [String: Any], defaultValue: T) -> Observable<T> {
        let fullPath = "https://kauth.kakao.com/oauth/token"
        print(fullPath)
        return RxAlamofire.data(.get, fullPath, parameters: parameters, encoding: URLEncoding.default, headers: ["Content-Type": "application/x-www-form-urlencoded; charset=utf-8"])
            .map { data -> T in
                return try JSONDecoder().decode(T.self, from:data)
            }
            .catchError { error in
                    print("Error occurred: \(error)")
                    return Observable.just(defaultValue) // 기본값 반환
            }
            .observeOn(self.queue)
            .debug()
    }
}
