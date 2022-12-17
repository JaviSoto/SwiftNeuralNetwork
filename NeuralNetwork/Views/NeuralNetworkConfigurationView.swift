//
//  File.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import SwiftUI

struct NeuralNetworkConfigurationView: View {
    let trainingData: MNISTData

    @ObservedObject
    private(set) var viewModel: NeuralNetworkViewModel

    var body: some View {
        VStack {
            List {
                Section {
                    GroupBox(label: SwiftUI.Label("Training", systemImage: "gear").font(.title2)) {
                        VStack(alignment: .leading) {
                            Text("Data Set:").bold()
                            Text("\(trainingData.training.items.count) training samples")
                            Text("\(trainingData.testing.items.count) test samples")
                            Divider()

                            let minMaxItems = min(viewModel.neuralNetwork.trainingData.items.count.double - 1, 500)
                            let maxMaxItems = viewModel.neuralNetwork.trainingData.items.count.double

                            ValueSlider(name: "Max items", value: $viewModel.neuralNetwork.configuration.maxTrainingItems.double, range: minMaxItems...maxMaxItems, step: max(1, ((maxMaxItems - minMaxItems) / 20).rounded()), decimalPoints: 0)

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
    }
}
