//
//  NeuralNetworkView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import SwiftUI

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
                                    let isOutputLayer = index == viewModel.neuralNetwork.configuration.layers.count - 1

                                    HStack {
                                        Text("\(index + 1):")
                                            .bold()

                                        if isOutputLayer {
                                            Text("Output Layer")
                                        } else {
                                            Text("Hidden Layer")
                                        }

                                        if viewModel.neuralNetwork.configuration.layers.count > 1 && !isOutputLayer {
                                            Button(action: {
                                                viewModel.neuralNetwork.configuration.layers.remove(at: index)
                                            }, label: {
                                                Image(systemName: "minus.circle.fill")
                                            })
                                        }
                                    }

                                    if !isOutputLayer {
                                        ValueSlider(name: "Number of neurons", value: $viewModel.neuralNetwork.configuration.layers[index].neuronCount.double, range: 1...20, step: 1, decimalPoints: 0)
                                    }

                                    Divider()
                                }
                            }

                            Button("Add layer") {
                                viewModel.neuralNetwork.configuration.layers.insert(.init(neuronCount: 10), at: viewModel.neuralNetwork.configuration.layers.count - 1)
                            }
                        }
                    }

                    Section {
                        GroupBox(label: SwiftUI.Label("Accuracy Evolution", systemImage: "chart.line.uptrend.xyaxis").font(.title2)) {
                            LearningAccuracyEvolutionGraph(sessions: viewModel.trainingSessionsAccuracies)
                                .padding()
                        }
                    }
                }

                Spacer()

                Divider()
                Text("Training Data Set Accuracy: \(viewModel.trainingDataAccuracy.formatted(.percent.precision(.significantDigits(3))))")
                Text("Test Data Set Accuracy: \(viewModel.testDataAccuracy.formatted(.percent.precision(.significantDigits(3))))")

                HStack(spacing: 10) {
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
                                .font(.title)

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

struct NeuralNetworkView_Previews: PreviewProvider {
    private static let imageWidth: UInt32 = 28

    private static var randomItem: MNISTParser.DataSet.Item {
        return (
            image: .init(pixels: Array(0..<(imageWidth * imageWidth)).map { _ in UInt8.random(in: 0...UInt8.max) }),
            label: .init(representedNumber: (UInt8(0)...9).randomElement()!)
        )
    }

    static var previews: some View {
        NeuralNetworkView(trainingData: .init(
            training: .init(imageWidth: imageWidth, items: (0..<10).map { _ in randomItem }),
            testing: .init(imageWidth: imageWidth, items: (0..<10).map { _ in randomItem }),
            all: .init(imageWidth: imageWidth, items: (0..<20).map { _ in randomItem })
        ))
    }
}
