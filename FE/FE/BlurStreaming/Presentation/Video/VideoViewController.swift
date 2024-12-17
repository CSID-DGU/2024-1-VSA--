//
//  VideoViewController.swift
//  BlurStreaming
//
//  Created by 신예빈 on 12/15/24.
//

import Foundation
import UIKit
import RxSwift
import SnapKit

fileprivate enum Section {
    case list
}

fileprivate enum Item: Hashable {
    case video(String, Int)
}

class VideoViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    var selectedVideoURL: URL?
    let disposeBag = DisposeBag()
    
    let viewModel = VideoViewModel()
    private var dataSource: UICollectionViewDiffableDataSource<Section, Item>?
    lazy var collectionView: UICollectionView = {
        let collectionView = UICollectionView(frame: .zero, collectionViewLayout: self.createLayout())
        collectionView.register(VideoCollectionViewCell.self, forCellWithReuseIdentifier: VideoCollectionViewCell.id)
        return collectionView
    } ()
    
    let updateTrigger = PublishSubject<Void>()
    
    private let titleLabel: UILabel = {
        let label = UILabel()
        label.text = "Video"
        label.textAlignment = .left
        label.numberOfLines = 0
        label.contentMode = .bottom
        label.font = .systemFont(ofSize: 28, weight: .medium)
        label.textColor = .black.withAlphaComponent(0.6)
        return label
    } ()
    
    
    private func setUI() {
        view.addSubview(titleLabel)
        view.addSubview(collectionView)
        
        titleLabel.snp.makeConstraints { make in
            make.top.equalToSuperview().offset(10)
            make.leading.equalToSuperview().offset(20)
        }
        
        collectionView.backgroundColor = .background
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setUI()
        setDataSource()
        bindViewModel()
        
        
        view.backgroundColor = .background
        
        let selectButton = UIButton(type: .system)
        selectButton.setTitle("동영상 선택 및 업로드", for: .normal)
        selectButton.addTarget(self, action: #selector(selectVideo), for: .touchUpInside)
        selectButton.frame = CGRect(x: 100, y: 200, width: 200, height: 50)
        view.addSubview(selectButton)
        
        
        let downloadButton = UIButton(type: .system)
        downloadButton.setTitle("동영상 다운로드", for: .normal)
        downloadButton.addTarget(self, action: #selector(downloadAndSaveVideo), for: .touchUpInside)
        downloadButton.frame = CGRect(x: 50, y: 200, width: 200, height: 50)
        view.addSubview(downloadButton)
        
        selectButton.snp.makeConstraints { make in
            make.top.equalTo(titleLabel.snp.bottom).offset(10)
            make.centerX.equalToSuperview()
        }
        
        collectionView.snp.makeConstraints { make in
            make.top.equalTo(selectButton.snp.bottom).offset(10)
            make.leading.equalToSuperview()
            make.width.equalToSuperview()
            make.height.equalTo(500)
        }
        
        downloadButton.snp.makeConstraints { make in
            make.top.equalTo(collectionView.snp.bottom).offset(30)
            make.centerX.equalToSuperview()
        }
        
        PhotoLibraryManager.shared.requestPhotoLibraryAccess { isAuthorized in
            if !isAuthorized {
                DispatchQueue.main.async {
                    let alert = UIAlertController(
                        title: "사진 접근 권한 필요",
                        message: "사진 앱에 접근하려면 설정에서 권한을 허용해주세요.",
                        preferredStyle: .alert
                    )
                    alert.addAction(UIAlertAction(title: "취소", style: .cancel, handler: nil))
                    alert.addAction(UIAlertAction(title: "설정으로 이동", style: .default) { _ in
                        self.openAppSettings()
                    })
                    self.present(alert, animated: true, completion: nil)
                }
            }
        }
        
    }
    
    @objc func selectVideo() {
            let picker = UIImagePickerController()
            picker.delegate = self
            picker.sourceType = .photoLibrary
            picker.mediaTypes = ["public.movie"] // 동영상만 선택 가능
            present(picker, animated: true)
        }

        // 동영상 선택 완료 시 호출
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            picker.dismiss(animated: true)

            if let videoURL = info[.mediaURL] as? URL {
                print("선택된 동영상: \(videoURL)")
                selectedVideoURL = videoURL
                uploadVideo(videoURL: videoURL)
            }
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
            print("동영상 선택 취소")
        }

        func uploadVideo(videoURL: URL) {
            guard let serverURL = URL(string: "http://192.168.219.102:8000/testapp/api/upload") else { return }

            let videoUploader = VideoUploader()
            videoUploader.uploadVideo(videoURL: videoURL, to: serverURL)
                .subscribe(onNext: { progress in
                    print("업로드 진행: \(progress * 100)%")
                }, onError: { error in
                    print("업로드 실패: \(error.localizedDescription)")
                }, onCompleted: {
                    print("업로드 완료!")
                    self.updateTrigger.onNext(())
                })
                .disposed(by: disposeBag)
        }
    
    @objc func downloadAndSaveVideo() {
            guard let videoURL = URL(string: "http://192.168.219.102:8000/testapp/api/download/file1.MOV") else {
                print("잘못된 URL")
                return
            }

            // 다운로드 경로 설정 (Documents 디렉토리)
            let fileManager = FileManager.default
            let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
            let destinationURL = documentsDirectory.appendingPathComponent(videoURL.lastPathComponent)

            // 1. 동영상 다운로드
            VideoDownloader.shared.downloadVideo(from: videoURL, to: destinationURL)
            .flatMap { localURL -> Observable<Void> in
                    print("동영상 다운로드 성공: \(localURL)")
                    // 2. 사진 앱에 저장
                    return PhotoLibraryManager.shared.saveVideoToPhotoLibrary(videoURL: localURL)
                }
            .observeOn(MainScheduler.instance).subscribe(onNext: {
                    print("동영상 사진 앱에 저장 완료!")
                }, onError: { error in
                    print("오류 발생: \(error.localizedDescription)")
                }, onCompleted: {
                    // 3. 동영상 재생
                    DispatchQueue.main.async {
                        VideoPlayerManager.playVideo(from: destinationURL, on: self)
                    }
                })
                .disposed(by: disposeBag)
        }
    
    func openAppSettings() {
        if let appSettings = URL(string: UIApplication.openSettingsURLString) {
            UIApplication.shared.open(appSettings, options: [:], completionHandler: nil)
        }
    }
    
    private func bindViewModel() {
        let input = VideoViewModel.Input(updateTrigger: updateTrigger)
        let out = viewModel.transform(input: input)
        
        
        out.video.bind {[weak self] response in
            var snapshot = NSDiffableDataSourceSnapshot<Section, Item>()
            
            let videoItems = response.data.videos.enumerated().map { index, item in
                Item.video(item, index+1) }
            let section = Section.list
            snapshot.appendSections([section])
            snapshot.appendItems(videoItems, toSection: section)
            self?.dataSource?.apply(snapshot)
        }.disposed(by: disposeBag)
    }
    
    private func createLayout() -> UICollectionViewCompositionalLayout {
        let config = UICollectionViewCompositionalLayoutConfiguration()
        config.interSectionSpacing = 10
        
        return UICollectionViewCompositionalLayout(sectionProvider: {[weak self] sectionIndex, _ in
            let section = self?.dataSource?.sectionIdentifier(for: sectionIndex)
            
            switch section {
            case .list:
                return self?.createVideoSection()
            default:
                return self?.createVideoSection()
            }
        }, configuration: config)
    }
    
    private func createVideoSection() -> NSCollectionLayoutSection {
        let itemSize = NSCollectionLayoutSize(widthDimension: .fractionalWidth(1.0), heightDimension: .absolute(60))
        let item = NSCollectionLayoutItem(layoutSize: itemSize)
        
        let groupSize = NSCollectionLayoutSize(widthDimension: .fractionalWidth(1.0), heightDimension: .absolute(75))
        let group = NSCollectionLayoutGroup.horizontal(layoutSize: groupSize, subitems: [item])
        
        let section = NSCollectionLayoutSection(group: group)
        section.contentInsets = .init(top: 30, leading: 30, bottom: 0, trailing: 30)
        return section
    }
    
    private func setDataSource() {
        dataSource = UICollectionViewDiffableDataSource<Section, Item>(collectionView: collectionView, cellProvider: { collectionView, indexPath, itemIdentifier in
            switch itemIdentifier {
            case .video(let itemData, let Index):
                let cell = collectionView.dequeueReusableCell(withReuseIdentifier: VideoCollectionViewCell.id, for: indexPath) as? VideoCollectionViewCell
                cell?.configure(title: itemData, index: Index)
                return cell!
            }
        })
    }
    
    override func viewWillAppear(_ animated: Bool) {
        updateTrigger.onNext(())
    }
}
