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
    var neuralNetwork: ImageRecognitionNeuralNetwork

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

    func predictions(forImage image: SampleImage) -> ImageRecognitionNeuralNetwork.PredictionOutcome {
        return neuralNetwork.digitPredictions(withInputImage: image)
    }

    private func updateAccuracies() {
        measure("Calculating accuracy") {
            trainingDataAccuracy = neuralNetwork.neuralNetwork.accuracy(usingInputData: trainingInputAndValidationMatrixes.input, expectedOutput: trainingInputAndValidationMatrixes.validation)
            testDataAccuracy = neuralNetwork.neuralNetwork.accuracy(usingInputData: testingInputAndValidationMatrixes.input, expectedOutput: testingInputAndValidationMatrixes.validation)
        }
    }
}
