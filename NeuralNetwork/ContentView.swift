//
//  ContentView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/14/22.
//

import SwiftUI

struct ContentView: View {
    enum Tab: CaseIterable {
        case images
        case training
    }

    @State
    var trainingData: MNISTParser.DataSet?

    @State
    var randomItem: MNISTParser.DataSet.Item?

    @State
    var neuralNetwork: ImageRecognitionNeuralNetwork?

    var body: some View {
        TabView {
            ForEach(Tab.allCases, id: \.self) {
                switch $0 {
                case .images:
                    if let trainingData, let randomItem {
                        SampleImageView(sampleImage: randomItem.image, width: trainingData.imageWidth)
                            .frame(width: 300)

                        Text("\(randomItem.label.representedNumber)")
                    }

                    Button("New image") {
                        updateImage()
                    }
                case .training:
                    if let trainingData {
                        Button("Train neural network") {
                            createNeuralNetwork(with: trainingData)
                        }
                    } else {
                        ProgressView()
                    }
                }
            }
        }
        .padding()
        .task {
            await measure("Loading data") {
                trainingData = await loadTrainingData()
                updateImage()
            }
        }
    }

    func updateImage() {
        randomItem = trainingData?.items.randomElement()!
    }

    func createNeuralNetwork(with trainingData: MNISTParser.DataSet) {
        self.neuralNetwork = ImageRecognitionNeuralNetwork(trainingData: trainingData)
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

private func loadTrainingData() async -> MNISTParser.DataSet {
    return await Task.detached { () -> MNISTParser.DataSet in
        let imageSetFile = Bundle.main.url(forResource: "train-images-idx3-ubyte", withExtension: nil)!
        let labelsFile = Bundle.main.url(forResource: "train-labels-idx1-ubyte", withExtension: nil)!

        return try! MNISTParser.loadData(imageSetFileURL: imageSetFile, labelDataFileURL: labelsFile)
    }.value
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
