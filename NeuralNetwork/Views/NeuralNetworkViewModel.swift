//
//  NeuralNetworkViewModel.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import Foundation

@MainActor
final class NeuralNetworkViewModel: ObservableObject {
    private let trainingData: MNISTData

    private let trainingInputAndValidationMatrixes: MNISTParser.DataSet.InputAndValidationMatrixes
    private let testingInputAndValidationMatrixes: MNISTParser.DataSet.InputAndValidationMatrixes

    @Published
    var neuralNetwork: ImageRecognitionNeuralNetwork {
        didSet {
            updateLayerState()
        }
    }

    enum State: Equatable {
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

    @Published
    private(set) var trainingSessionsAccuracies: [NeuralNetwork.TrainingSessionAccuracyProgress] = []

    @Published
    private(set) var trainingLayerState: [NeuralNetwork.TrainingProgressObserver.LayerState] = []

    private let trainingProgressObserver = NeuralNetwork.TrainingProgressObserver()

    init(trainingData: MNISTData) {
        self.trainingData = trainingData

        self.trainingInputAndValidationMatrixes = trainingData.training.inputAndValidationMatrixes
        self.testingInputAndValidationMatrixes = trainingData.testing.inputAndValidationMatrixes

        self.neuralNetwork = ImageRecognitionNeuralNetwork(trainingData: trainingData.training)
        self.updateAccuracies()
        self.updateLayerState()
    }

    func train() {
        self.state = .training

        let trainingSessionIndex = trainingSessionsAccuracies.count
        trainingSessionsAccuracies.append([])

        let token = trainingProgressObserver.$accuracies.combineLatest(trainingProgressObserver.$layerState)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] (accuracies, layerState) in
                self?.trainingSessionsAccuracies[trainingSessionIndex] = accuracies
                self?.trainingLayerState = layerState
            }

        Task {
            var neuralNetwork = neuralNetwork

            await neuralNetwork.trainAsync(with: trainingProgressObserver)

            await MainActor.run { [neuralNetwork] in
                self.neuralNetwork = neuralNetwork
                self.updateAccuracies()
                self.state = .trained

                token.cancel()
            }
        }
    }

    func stopTraining() {
        trainingProgressObserver.shouldStopTraining.value = true
    }

    func predictions(forImage image: SampleImage) -> ImageRecognitionNeuralNetwork.PredictionOutcome {
        return neuralNetwork.digitPredictions(withInputImage: image)
    }

    private func updateAccuracies() {
        measure("Calculating accuracy") {
            final class CalculationResult: @unchecked Sendable {
                var trainingDataAccuracy: Double = 0
                var testDataAccuracy: Double = 0
            }

            let result = CalculationResult()

            DispatchQueue.concurrentPerform(iterations: 2) { [neuralNetwork = neuralNetwork.neuralNetwork!] index in
                switch index {
                case 0:
                    result.trainingDataAccuracy = neuralNetwork.accuracy(usingInputData: trainingInputAndValidationMatrixes.input, expectedOutput: trainingInputAndValidationMatrixes.validation)
                case 1:
                    result.testDataAccuracy = neuralNetwork.accuracy(usingInputData: testingInputAndValidationMatrixes.input, expectedOutput: testingInputAndValidationMatrixes.validation)
                default: fatalError()
                }
            }

            self.trainingDataAccuracy = result.trainingDataAccuracy
            self.testDataAccuracy = result.testDataAccuracy
        }
    }

    private func updateLayerState() {
        let forwardPropagation = self.trainingLayerState.map(\.layer.neuronCount) == neuralNetwork.neuralNetwork.layers.map(\.neuronCount)
        ? self.trainingLayerState.map(\.forwardPropagation)
        : []

        self.trainingLayerState = neuralNetwork.neuralNetwork.layers.enumerated().map { (index, layer) in
                .init(layer: layer, forwardPropagation: forwardPropagation.isEmpty ? nil : forwardPropagation[index])
        }
    }
}
