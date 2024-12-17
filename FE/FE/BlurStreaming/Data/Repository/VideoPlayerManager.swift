//
//  VideoPlayerManager.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/16/24.
//

import Foundation
import AVKit

class VideoPlayerManager {
    static func playVideo(from url: URL, on viewController: UIViewController) {
        let player = AVPlayer(url: url)
        let playerViewController = AVPlayerViewController()
        playerViewController.player = player

        viewController.present(playerViewController, animated: true) {
            player.play()
        }
    }
}
