//
//  MyPageViewController.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/17/24.
//

import Foundation
import UIKit
import SnapKit

class MyPageViewController: UIViewController {
    
    private let titleLabel: UILabel = {
        let label = UILabel()
        label.text = "Mypage"
        label.textAlignment = .left
        label.numberOfLines = 0
        label.contentMode = .bottom
        label.font = .systemFont(ofSize: 28, weight: .medium)
        label.textColor = .black.withAlphaComponent(0.6)
        return label
    } ()
    
    private let nicknameView: UIView = {
        let view = UIView()
        view.backgroundColor = .white
        return view
    } ()
    
    private let nicknameLabel: UILabel = {
        let label = UILabel()
        label.text = "닉네임 설정"
        label.textAlignment = .left
        label.numberOfLines = 0
        label.contentMode = .bottom
        label.font = .systemFont(ofSize: 15, weight: .medium)
        label.textColor = .black.withAlphaComponent(0.6)
        return label
    } ()
    
    private let photoView: UIView = {
        let view = UIView()
        view.backgroundColor = .white
        return view
    } ()
    
    private let photoLabel: UILabel = {
        let label = UILabel()
        label.text = "사진 등록"
        label.textAlignment = .left
        label.numberOfLines = 0
        label.contentMode = .bottom
        label.font = .systemFont(ofSize: 15, weight: .medium)
        label.textColor = .black.withAlphaComponent(0.6)
        return label
    } ()
    
    private func setUI() {
        self.view.backgroundColor = .background
        
        self.view.addSubview(titleLabel)
        self.view.addSubview(nicknameView)
        self.view.addSubview(photoView)
        
        nicknameView.addSubview(nicknameLabel)
        
        photoView.addSubview(photoLabel)
        
        titleLabel.snp.makeConstraints { make in
            make.top.equalToSuperview().offset(10)
            make.leading.equalToSuperview().offset(30)
        }
        
        nicknameView.snp.makeConstraints { make in
            make.top.equalTo(titleLabel.snp.bottom).offset(10)
            make.leading.equalToSuperview().offset(30)
            make.trailing.equalToSuperview().offset(-30)
            make.height.equalTo(150)
        }
        
        photoView.snp.makeConstraints { make in
            make.top.equalTo(nicknameView.snp.bottom).offset(10)
            make.leading.equalToSuperview().offset(30)
            make.trailing.equalToSuperview().offset(-30)
            make.height.equalTo(300)
        }
        
        nicknameLabel.snp.makeConstraints { make in
            make.top.equalToSuperview().offset(10)
            make.leading.equalToSuperview().offset(10)
        }
        
        photoLabel.snp.makeConstraints { make in
            make.top.equalToSuperview().offset(10)
            make.leading.equalToSuperview().offset(10)
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setUI()
        
        
        
    }
}
