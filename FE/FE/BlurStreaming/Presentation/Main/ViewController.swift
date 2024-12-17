//
//  ViewController.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/15/24.
//

import UIKit
import SnapKit
import RxSwift

class ViewController: UITabBarController {

    
    
    private func setUI() {
        
        
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setUI()
        setAttribute()
        
        self.selectedIndex = 1
        view.backgroundColor = .white
        // Do any additional setup after loading the view.
    }

    let videoViewController = VideoViewController()
    let streamingViewController = StreamingViewController()
    let myPageViewController = MyPageViewController()
    
    private func setAttribute() {
        viewControllers = [
            createNavigationController(for: videoViewController, title: "Video", image: UIImage(named: "video")!, selectedImage: UIImage(named: "video")!),
            createNavigationController(for: streamingViewController, title: "Live", image: UIImage(named: "live")!, selectedImage: UIImage(named: "live")!),
            createNavigationController(for: myPageViewController, title: "Mypage", image: UIImage(named: "mypage")!, selectedImage: UIImage(named: "mypage")!)
        ]
    }

    fileprivate func createNavigationController(for rootViewController: UIViewController, title: String?, image: UIImage, selectedImage: UIImage) -> UIViewController {
            let navigationController = UINavigationController(rootViewController:  rootViewController)
            navigationController.navigationBar.isTranslucent = false
            navigationController.navigationBar.backgroundColor = .white
            navigationController.tabBarItem.title = title
            navigationController.tabBarItem.image = image
            navigationController.tabBarItem.selectedImage = selectedImage
            return navigationController
        }
}

