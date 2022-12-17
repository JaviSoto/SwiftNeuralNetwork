//
//  ImageRecognitionNeuralNetwork.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/15/22.
//

import Foundation
import SwiftMatrix

struct ImageRecognitionNeuralNetwork {
    private(set) var neuralNetwork: NeuralNetwork!

    var configuration = Configuration()

    let trainingData: MNISTParser.DataSet

    struct Configuration {
        struct LayerConfig {
            var neuronCount: Int
        }

        var maxTrainingItems: Int = 10000
        var iterations: Int = 300
        var learningRate: Double = 0.03

        var layers: [LayerConfig] = [.init(neuronCount: 10), .init(neuronCount: 10)]
    }

    init(trainingData: MNISTParser.DataSet) {
        self.trainingData = trainingData

        resetNeuralNetwork()
    }

    mutating func train(with observer: NeuralNetwork.TrainingProgressObserver) {
        resetNeuralNetwork()

        let (training, validation) = trainingData
            .shuffle()
            .cropped(maxLength: configuration.maxTrainingItems)
            .trainingAndValidationMatrixes

        neuralNetwork.train(
            usingTrainingData: training,
            validationData: validation,
            iterations: configuration.iterations,
            learningRate: configuration.learningRate,
            progressObserver: observer
        )
    }

    mutating func trainAsync(with observer: NeuralNetwork.TrainingProgressObserver) async {
        let copy = self

        let trained = await Task.detached { () -> ImageRecognitionNeuralNetwork in
            return measure("Training NN") { [copy] in
                var copy = copy
                copy.train(with: observer)
                return copy
            }
        }.value

        self = trained
    }

    struct PredictionOutcome {
        struct Digit: Equatable, Identifiable {
            let value: Int
            let confidence: Double

            var id: Int {
                return value
            }
        }

        init() {
            self.digits = (0...9).map { Digit(value: $0, confidence: 0) }
        }

        init(digits: [Digit]) {
            self.digits = digits
        }

        var digits: [Digit] {
            didSet {
                assert(digits.count == 10)
            }
        }

        var highestDigit: Digit {
            return digits.max(by: { $1.confidence > $0.confidence })!
        }
    }

    func digitPredictions(withInputImage image: SampleImage) -> PredictionOutcome {
        let predictionMatrix = neuralNetwork.predictions(usingData: image.normalizedPixelVector)

        return PredictionOutcome(digits: predictionMatrix′.mutableValues.enumerated().map { .init(value: $0, confidence: $1) })
    }

    // MARK: - Private

    private mutating func resetNeuralNetwork() {
        neuralNetwork = NeuralNetwork(
            inputLayerNeuronCount: Int(trainingData.imageWidth * trainingData.imageWidth),
            outputLayerSize: 10
        )

        for (index, layer) in configuration.layers.enumerated() {
            let isOutputLayer = index == configuration.layers.count - 1
            neuralNetwork.addHiddenLayer(withNeuronCount: layer.neuronCount, activationFunction: isOutputLayer ? .softMax : .reLU)
        }
    }
}

private extension SampleImage {
    var normalizedPixelVector: [Double] {
        return pixels.map { Double($0) / Double(SampleImage.Pixel.max) }
    }
}

extension MNISTParser.DataSet {
    var trainingAndValidationMatrixes: (training: Matrix, validation: Matrix) {
        let training = Matrix(self.items.map { $0.image.normalizedPixelVector })′
        let validation = Matrix(self.items.map { [Double($0.label.representedNumber)] })

        return (training, validation)
    }
}

private extension Matrix {
    init(_ sampleSet: MNISTParser.DataSet) {
        self = Matrix(sampleSet.items.map { item in
            item.image.normalizedPixelVector
        })
    }
}
