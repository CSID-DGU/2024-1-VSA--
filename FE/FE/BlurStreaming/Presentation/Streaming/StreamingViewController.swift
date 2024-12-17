//
//  StreamingViewController.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/17/24.
//

import Foundation
import UIKit
import HaishinKit
import SRTHaishinKit
import AVFoundation

class StreamingViewController: UIViewController {
    
    private let topView: UIView = {
        let view = UIView()
        view.backgroundColor = .white
        return view
    } ()
    
    private let titleLabel: UILabel = {
        let label = UILabel()
        label.text = "Blureaming"
        label.textAlignment = .left
        label.numberOfLines = 0
        label.contentMode = .bottom
        label.font = .systemFont(ofSize: 28, weight: .medium)
        label.textColor = .black.withAlphaComponent(0.6)
        return label
    } ()
    
    let alarmButton: UIButton = {
        let button = UIButton()
        button.setImage(UIImage(named: "alarm"), for: .normal)
        return button
    }()
    
    private func setUI() {
        view.backgroundColor = .background
        
        view.addSubview(topView)
        topView.addSubview(titleLabel)
        topView.addSubview(alarmButton)
        
        topView.snp.makeConstraints { make in
            make.top.equalToSuperview()
            make.leading.trailing.equalToSuperview()
            make.height.equalTo(50)
        }
        
        titleLabel.snp.makeConstraints { make in
            make.bottom.equalToSuperview().offset(-10)
            make.leading.equalToSuperview().offset(20)
        }
        
        alarmButton.snp.makeConstraints { make in
            make.bottom.equalToSuperview().offset(-10)
            make.trailing.equalToSuperview().offset(-20)
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setUI()
    }
    
}
