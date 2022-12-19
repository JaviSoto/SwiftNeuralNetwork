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

    @State
    var neuronVisualization: LayerStatusView.Visualization = .weights

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

                            ValueSlider(name: "Iteration Batch Size", value: $viewModel.neuralNetwork.configuration.maxTrainingItems.double, range: minMaxItems...maxMaxItems, step: max(1, ((maxMaxItems - minMaxItems) / 100).rounded()), decimalPoints: 0)

                            ValueSlider(name: "Iterations", value: $viewModel.neuralNetwork.configuration.iterations.double, range: 1...3500, step: 100, decimalPoints: 0)

                            ValueSlider(name: "Learning Rate", value: $viewModel.neuralNetwork.configuration.learningRate, range: 0.01...1, step: 0.05, decimalPoints: 2)
                        }
                    }
                }

                Section {
                    GroupBox(label: SwiftUI.Label("Layers", systemImage: "square.2.layers.3d").font(.title2)) {
                        ForEach(viewModel.neuralNetwork.configuration.layers.indexed) { layerConfig in
                            VStack {
                                let isOutputLayer = layerConfig.index == viewModel.neuralNetwork.configuration.layers.count - 1

                                HStack {
                                    Text("\(layerConfig.index + 1):")
                                        .bold()

                                    if isOutputLayer {
                                        Text("Output Layer")
                                    } else {
                                        Text("Hidden Layer")
                                    }

                                    if viewModel.neuralNetwork.configuration.layers.count > 1 && !isOutputLayer {
                                        Button(action: {
                                            viewModel.neuralNetwork.configuration.layers.remove(at: layerConfig.index)
                                        }, label: {
                                            Image(systemName: "minus.circle.fill")
                                        })
                                    }
                                }

                                if isOutputLayer {
                                    Text("\(viewModel.neuralNetwork.neuralNetwork.outputLayerSize) output neurons")
                                        .italic()
                                        .frame(alignment: .leading)
                                } else {
                                    ValueSlider(name: "Number of neurons", value: $viewModel.neuralNetwork.configuration.layers[layerConfig.index].neuronCount.double, range: 1...20, step: 1, decimalPoints: 0)
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
                            .frame(height: 200)
                            .padding()
                    }
                }

                Section {
                    GroupBox(label: SwiftUI.Label("Layer Visualization", systemImage: "eye.fill").font(.title2)) {
                        Picker("Style", selection: $neuronVisualization) {
                            ForEach(LayerStatusView.Visualization.allCases, id: \.self) { visualization in
                                Text("\(visualization.name)")
                            }
                        }
                        .frame(width: 200)

                        LayerStatusView(layers: viewModel.trainingLayerState, visualization: neuronVisualization)
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
                    Button("Stop training session") {
                        viewModel.stopTraining()
                    }

                    ProgressView()
                } else {
                    Button("Train neural network") {
                        viewModel.train()
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .padding(.vertical)
        .navigationTitle("Configuration")
    }
}
