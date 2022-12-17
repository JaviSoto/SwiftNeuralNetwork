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
            trainingData = await loadTrainingData()
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

    @Published
    private(set) var trainingDataAccuracy: Double = 0

    @Published
    private(set) var testDataAccuracy: Double = 0

    init(trainingData: MNISTData) {
        self.trainingData = trainingData
        self.neuralNetwork = ImageRecognitionNeuralNetwork(trainingData: trainingData.training)
        self.updateAccuracies()
    }

    func train() {
        self.state = .training

        Task {
            var neuralNetwork = neuralNetwork

            await neuralNetwork.trainAsync()

            await MainActor.run { [neuralNetwork] in
                self.neuralNetwork = neuralNetwork
                self.updateAccuracies()
                self.state = .trained
            }
        }
    }

    func predictions(forImage image: SampleImage) -> ImageRecognitionNeuralNetwork.PredictionOutcome {
        return measure("Calculating predictions") {
            return neuralNetwork.digitPredictions(withInputImage: image)
        }
    }

    private func updateAccuracies() {
        measure("Calculating accuracy") {
            let (trainingImages, trainingLabels) = trainingData.training.trainingAndValidationMatrixes
            trainingDataAccuracy = neuralNetwork.neuralNetwork.accuracy(usingInputData: trainingImages, expectedOutput: trainingLabels)

            let (testImages, testLabels) = trainingData.testing.trainingAndValidationMatrixes
            testDataAccuracy = neuralNetwork.neuralNetwork.accuracy(usingInputData: testImages, expectedOutput: testLabels)
        }
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

    @State
    private var randomItem: MNISTParser.DataSet.Item?

    @State
    private var randomItemPredictionOutcome: ImageRecognitionNeuralNetwork.PredictionOutcome = .init()

    @State
    private var predictionOutcomeTableOrder = [KeyPathComparator(\ImageRecognitionNeuralNetwork.PredictionOutcome.Digit.confidence, order: .reverse)]

    var body: some View {
        NavigationView {
            VStack {
                List {
                    Section {
                        GroupBox(label: SwiftUI.Label("Training", systemImage: "gear").font(.title2)) {
                            VStack(alignment: .leading) {
                                Text("Data Set:").bold()
                                Text("\(trainingData.training.items.count) training samples")
                                Text("\(trainingData.testing.items.count) test samples")
                                Divider()

                                ValueSlider(name: "Max items", value: $viewModel.neuralNetwork.configuration.maxTrainingItems.double, range: 500...viewModel.neuralNetwork.trainingData.items.count.double, step: 500, decimalPoints: 0)

                                ValueSlider(name: "Iterations", value: $viewModel.neuralNetwork.configuration.iterations.double, range: 1...1500, step: 100, decimalPoints: 0)

                                ValueSlider(name: "Learning Rate", value: $viewModel.neuralNetwork.configuration.learningRate, range: 0.01...1, step: 0.05, decimalPoints: 2)

                            }
                        }
                    }

                    Section {
                        GroupBox(label: SwiftUI.Label("Layers", systemImage: "square.2.layers.3d").font(.title2)) {
                            ForEach(Array(viewModel.neuralNetwork.configuration.layers.enumerated()), id: \.0) { (index, layerConfig) in
                                VStack {
                                    HStack {
                                        Text("Layer \(index + 1)")

                                        if viewModel.neuralNetwork.configuration.layers.count > 1 {
                                            Button(action: {
                                                viewModel.neuralNetwork.configuration.layers.remove(at: index)
                                            }, label: {
                                                Image(systemName: "minus.circle.fill")
                                            })
                                        }
                                    }

                                    ValueSlider(name: "Number of neurons", value: $viewModel.neuralNetwork.configuration.layers[index].neuronCount.double, range: 1...20, step: 1, decimalPoints: 0)

                                    Divider()
                                }
                            }

                            Button("Add layer") {
                                viewModel.neuralNetwork.configuration.layers.append(.init(neuronCount: 10))
                            }
                        }
                    }
                }

                Spacer()

                Divider()
                Text("Training Data Set Accuracy: \(viewModel.trainingDataAccuracy.formatted(.percent.precision(.significantDigits(3))))")
                Text("Test Data Set Accuracy: \(viewModel.testDataAccuracy.formatted(.percent.precision(.significantDigits(3))))")

                HStack {
                    if viewModel.state == .training {
                        ProgressView()
                    }
                    Button("Train neural network") {
                        viewModel.train()
                    }
                    .disabled(viewModel.state == .training)
                }
            }
            .padding(.vertical)
            .frame(minWidth: 300)
            .navigationTitle("Configuration")

            VStack {
                if let randomItem {
                    VStack {
                        VStack {
                            Text("Data label: \(randomItem.label.representedNumber)")
                                .font(.largeTitle)

                            SampleImageView(sampleImage: randomItem.image, width: trainingData.training.imageWidth)
                                .frame(width: 300)
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
                        .frame(height: 300)
                    }
                }

                Spacer()

                Divider()

                Button("Random image") {
                    updateImage()
                }
            }
            .padding(.vertical)
        }
        .navigationSplitViewStyle(.prominentDetail)
        .frame(minWidth: 1000, minHeight: 600)
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

struct MNISTData {
    let training: MNISTParser.DataSet
    let testing: MNISTParser.DataSet

    let all: MNISTParser.DataSet
}

private func loadTrainingData() async -> MNISTData {
    return await Task.detached { () -> MNISTData in
        return measure("Loading training data") {
            let trainingImages = Bundle.main.url(forResource: "train-images-idx3-ubyte", withExtension: nil)!
            let trainingLabels = Bundle.main.url(forResource: "train-labels-idx1-ubyte", withExtension: nil)!

            let testImages = Bundle.main.url(forResource: "t10k-images-idx3-ubyte", withExtension: nil)!
            let testLabels = Bundle.main.url(forResource: "t10k-labels-idx1-ubyte", withExtension: nil)!

#if DEBUG
            let maxCount = 1000
#else
            let maxCount: Int? = nil
#endif

            let training = try! MNISTParser.loadData(imageSetFileURL: trainingImages, labelDataFileURL: trainingLabels, maxCount: maxCount)
            let testing = try! MNISTParser.loadData(imageSetFileURL: testImages, labelDataFileURL: testLabels, maxCount: maxCount)

            return .init(training: training, testing: testing, all: training + testing)
        }
    }.value
}

struct ContentView_Previews: PreviewProvider     {
    static var previews: some View {
        ContentView()
    }
}
