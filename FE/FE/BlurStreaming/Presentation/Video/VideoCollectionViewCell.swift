//
//  VideoCollectionViewCell.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/17/24.
//

import Foundation
import UIKit
import SnapKit

final class VideoCollectionViewCell: UICollectionViewCell {
    static let id = "VideoCollectionViewCell"
    
    private let indexLabel: UILabel = {
        let label = UILabel()
        label.text = ""
        label.font = .systemFont(ofSize: 18, weight: .semibold)
        return label
    } ()
    
    private let titleLabel: UILabel = {
        let label = UILabel()
        label.text = ""
        label.font = .systemFont(ofSize: 18, weight: .semibold)
        return label
    } ()
    
    private let downloadButton: UIButton = {
        let button = UIButton()
        button.setImage(UIImage(named: "download"), for: .normal)
        return button
    } ()
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        
        self.addSubview(indexLabel)
        self.addSubview(titleLabel)
        self.addSubview(downloadButton)
        
        self.backgroundColor = .white
        
        self.layer.borderWidth = 1
        self.layer.borderColor = UIColor.lightGray.cgColor
        
        indexLabel.snp.makeConstraints { make in
            make.leading.equalToSuperview().offset(20)
            make.centerY.equalToSuperview()
        }
        
        titleLabel.snp.makeConstraints { make in
            make.leading.equalTo(indexLabel.snp.trailing).offset(20)
            make.centerY.equalToSuperview()
        }
        
        downloadButton.snp.makeConstraints { make in
            make.trailing.equalToSuperview().offset(-30)
            make.centerY.equalToSuperview()
        }
    }
    
    public func configure(title: String, index: Int) {
        titleLabel.text = title
        indexLabel.text = String(index)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}
