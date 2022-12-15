//
//  ContentView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/14/22.
//

import SwiftUI

struct ContentView: View {
    @State
    private var trainingData: MNISTParser.DataSet?

    var body: some View {
        Group {
            if let trainingData {
                NeuralNetworkView(trainingData: trainingData)
            } else {
                VStack {
                    Text("Loading training data")
                    ProgressView()
                }
            }
        }.task {
            await measure("Loading data") {
                trainingData = await loadTrainingData()
            }
        }
    }
}

@MainActor
private final class NeuralNetworkViewModel: ObservableObject {
    @Published
    private(set) var neuralNetwork: ImageRecognitionNeuralNetwork

    @Published
    private(set) var isTraining = false

    init(trainingData: MNISTParser.DataSet) {
        self.neuralNetwork = ImageRecognitionNeuralNetwork(trainingData: trainingData)
    }

    func train() {
        Task {
            var neuralNetwork = neuralNetwork

            await measure("Training NN") {
                await neuralNetwork.trainAsync()
            }

            await MainActor.run { [neuralNetwork] in
                self.neuralNetwork = neuralNetwork
            }
        }
    }
}

struct NeuralNetworkView: View {
    let trainingData: MNISTParser.DataSet

    @ObservedObject
    private var viewModel: NeuralNetworkViewModel

    init(trainingData: MNISTParser.DataSet) {
        self.trainingData = trainingData
        self.viewModel = NeuralNetworkViewModel(trainingData: trainingData)
    }

    enum Tab: CaseIterable {
        case training
        case images
    }

    @State
    private var randomItem: MNISTParser.DataSet.Item?

    private var imageDisplayTab: some View {
        VStack {
            if let randomItem {
                SampleImageView(sampleImage: randomItem.image, width: trainingData.imageWidth)
                    .frame(width: 300)

                Text("\(randomItem.label.representedNumber)")
            }

            Button("Random image") {
                updateImage()
            }
        }
    }

    private var trainingTab: some View {
        VStack {
            Button("Train neural network") {
                viewModel.train()
            }
            .disabled(viewModel.isTraining)
        }
    }

    var body: some View {
        Text("Loaded \(trainingData.items.count) samples")

        TabView {
            ForEach(Tab.allCases, id: \.self) {
                switch $0 {
                case .images:
                    imageDisplayTab
                        .tabItem { Text("Images") }
                case .training:
                    trainingTab
                        .tabItem { Text("Training") }
                }
            }
        }
        .padding()
        .onAppear {
            updateImage()
        }
    }

    func updateImage() {
        randomItem = trainingData.items.randomElement()!
    }
}

struct SampleImageView: View {
    let sampleImage: SampleImage
    let width: UInt32

    var body: some View {
        sampleImage.asSwiftUIImage(width: width)
            .resizable()
            .aspectRatio(1, contentMode: .fit)
    }
}

func measure<T>(_ name: String, _ f: () async -> T) async -> T {
    print("Starting '\(name)'")

    let start = CFAbsoluteTimeGetCurrent()
    let result = await f()
    let end = CFAbsoluteTimeGetCurrent()

    let diff = end - start
    let duration = diff < 1 ? "\(Int(diff * 1000))ms" : "\(diff)s"

    print("'\(name)' took \(duration)")

    return result
}

private extension ImageRecognitionNeuralNetwork {
    mutating func trainAsync() async {
        let copy = self

        let trained = await Task.detached { () -> ImageRecognitionNeuralNetwork in
            var copy = copy
            copy.train()
            return copy
        }.value

        self = trained
    }
}

private func loadTrainingData() async -> MNISTParser.DataSet {
    return await Task.detached { () -> MNISTParser.DataSet in
//        let imageSetFile = Bundle.main.url(forResource: "train-images-idx3-ubyte", withExtension: nil)!
//        let labelsFile = Bundle.main.url(forResource: "train-labels-idx1-ubyte", withExtension: nil)!

        let imageSetFile = Bundle.main.url(forResource: "t10k-images-idx3-ubyte", withExtension: nil)!
        let labelsFile = Bundle.main.url(forResource: "t10k-labels-idx1-ubyte", withExtension: nil)!

        return try! MNISTParser.loadData(imageSetFileURL: imageSetFile, labelDataFileURL: labelsFile)
    }.value
}

struct ContentView_Previews: PreviewProvider     {
    static var previews: some View {
        ContentView()
    }
}
