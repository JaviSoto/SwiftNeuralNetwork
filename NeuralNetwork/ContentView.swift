//
//  ContentView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/14/22.
//

import SwiftUI
import SwiftMatrix

struct ContentView: View {
    @State
    private var trainingData: MNISTData?

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
    private let trainingData: MNISTData
    @Published
    var neuralNetwork: ImageRecognitionNeuralNetwork

    enum State {
        case idle
        case training
        case trained
    }

    @Published
    private(set) var state: State = .idle

    init(trainingData: MNISTData) {
        self.trainingData = trainingData
        self.neuralNetwork = ImageRecognitionNeuralNetwork(trainingData: trainingData.training)
    }

    func train() {
        self.state = .training

        Task {
            var neuralNetwork = neuralNetwork

            await measure("Training NN") {
                await neuralNetwork.trainAsync()
            }

            await MainActor.run { [neuralNetwork] in
                self.neuralNetwork = neuralNetwork
                self.state = .trained
            }
        }
    }

    func predictions(forImage image: SampleImage) -> ImageRecognitionNeuralNetwork.PredictionOutcome {
        return neuralNetwork.digitPredictions(withInputImage: image)
    }

    func trainingDataSetAccuracy() -> Double {
        let (images, labels) = trainingData.training.trainingAndValidationMatrixes

        return neuralNetwork.neuralNetwork.accuracy(usingInputData: images, expectedOutput: labels)
    }

    func testDataSetAccuracy() -> Double {
        let (images, labels) = trainingData.testing.trainingAndValidationMatrixes

        return neuralNetwork.neuralNetwork.accuracy(usingInputData: images, expectedOutput: labels)
    }
}

struct NeuralNetworkView: View {
    let trainingData: MNISTData

    @ObservedObject
    private var viewModel: NeuralNetworkViewModel

    init(trainingData: MNISTData) {
        self.trainingData = trainingData
        self.viewModel = NeuralNetworkViewModel(trainingData: trainingData)
    }

    enum Tab: CaseIterable {
        case training
        case images
    }

    @State
    private var randomItem: MNISTParser.DataSet.Item?

    @State
    private var randomItemPredictionOutcome: ImageRecognitionNeuralNetwork.PredictionOutcome = .init()

    @State
    private var predictionOutcomeTableOrder = [KeyPathComparator(\ImageRecognitionNeuralNetwork.PredictionOutcome.Digit.confidence, order: .reverse)]

    private var imageDisplayTab: some View {
        VStack {
            if let randomItem {
                HStack {
                    VStack {
                        SampleImageView(sampleImage: randomItem.image, width: trainingData.training.imageWidth)
                            .frame(width: 300)

                        Text("Data label: \(randomItem.label.representedNumber)")
                    }

                    Table(randomItemPredictionOutcome.digits, sortOrder: $predictionOutcomeTableOrder) {
                        TableColumn("Digit") { digit in
                            Text("\(digit.value)")
                                .bold(randomItemPredictionOutcome.highestDigit.value == digit.value)
                        }
                        .width(50)

                        TableColumn("Confidence") { digit in
                            Text("\(digit.confidence.formatted(.percent.precision(.significantDigits(3))))")
                                .bold(randomItemPredictionOutcome.highestDigit.value == digit.value)
                        }
                    }
                    .onChange(of: predictionOutcomeTableOrder) { newOrder in
                        randomItemPredictionOutcome.digits.sort(using: newOrder)
                    }
                }
            }

            Button("Random image") {
                updateImage()
            }
        }
    }

    private var trainingTab: some View {
        HStack {
            VStack {
                HStack {
                    VStack {
                        Text("Configuration")
                            .font(.title2)

                        ValueSlider(name: "Max items", value: $viewModel.neuralNetwork.configuration.maxTrainingItems.double, range: 500...viewModel.neuralNetwork.trainingData.items.count.double, step: 500, decimalPoints: 0)

                        ValueSlider(name: "Iterations", value: $viewModel.neuralNetwork.configuration.iterations.double, range: 1...1500, step: 100, decimalPoints: 0)

                        ValueSlider(name: "Learning Rate", value: $viewModel.neuralNetwork.configuration.learningRate, range: 0.01...1, step: 0.05, decimalPoints: 2)

                        Text("Layers")
                        TabView {
                            ForEach(Array(viewModel.neuralNetwork.configuration.layers.enumerated()), id: \.0) { (index, layerConfig) in
                                VStack {
                                    ValueSlider(name: "Number of neurons", value: $viewModel.neuralNetwork.configuration.layers[index].neuronCount.double, range: 1...20, step: 1, decimalPoints: 0)
                                    if viewModel.neuralNetwork.configuration.layers.count > 1 {
                                        Button("Remove") {
                                            viewModel.neuralNetwork.configuration.layers.remove(at: index)
                                        }
                                    }
                                }
                                .tabItem { Text("Layer \(index + 1)") }
                            }
                        }

                        Button("Add layer") {
                            viewModel.neuralNetwork.configuration.layers.append(.init(neuronCount: 10))
                        }
                    }
                    Spacer()
                }
            }
            VStack {
                HStack {
                    if viewModel.state == .training {
                        ProgressView()
                    }
                    Button("Train neural network") {
                        viewModel.train()
                    }
                    .disabled(viewModel.state == .training)
                }

                Text("Training Data Set Accuracy: \(viewModel.trainingDataSetAccuracy().formatted(.percent.precision(.significantDigits(3))))")
                Text("Test Data Set Accuracy: \(viewModel.testDataSetAccuracy().formatted(.percent.precision(.significantDigits(3))))")
            }
        }
    }

    var body: some View {
        Text("Loaded \(trainingData.training.items.count + trainingData.testing.items.count) samples")

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
        let item = trainingData.all.items.randomElement()!
        randomItem = item
        randomItemPredictionOutcome = viewModel.predictions(forImage: item.image)
        randomItemPredictionOutcome.digits.sort(using: predictionOutcomeTableOrder)
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

struct MNISTData {
    let training: MNISTParser.DataSet
    let testing: MNISTParser.DataSet

    let all: MNISTParser.DataSet
}

private func loadTrainingData() async -> MNISTData {
    return await Task.detached { () -> MNISTData in
        let trainingImages = Bundle.main.url(forResource: "train-images-idx3-ubyte", withExtension: nil)!
        let trainingLabels = Bundle.main.url(forResource: "train-labels-idx1-ubyte", withExtension: nil)!

        let testImages = Bundle.main.url(forResource: "t10k-images-idx3-ubyte", withExtension: nil)!
        let testLabels = Bundle.main.url(forResource: "t10k-labels-idx1-ubyte", withExtension: nil)!

        let training = try! MNISTParser.loadData(imageSetFileURL: trainingImages, labelDataFileURL: trainingLabels)

        let testing = try! MNISTParser.loadData(imageSetFileURL: testImages, labelDataFileURL: testLabels)

        return .init(training: training, testing: testing, all: training + testing)
    }.value
}

struct ContentView_Previews: PreviewProvider     {
    static var previews: some View {
        ContentView()
    }
}
