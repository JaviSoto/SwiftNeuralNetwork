//
//  ImageRecognitionNeuralNetwork.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/15/22.
//

import Foundation
import SwiftMatrix

struct ImageRecognitionNeuralNetwork {
    private var neuralNetwork: NeuralNetwork

    private let trainingData: MNISTParser.DataSet

    init(trainingData: MNISTParser.DataSet) {
        self.trainingData = trainingData

        self.neuralNetwork = NeuralNetwork(inputLayerNeuronCount: Int(trainingData.imageWidth * trainingData.imageWidth))
        neuralNetwork.addHiddenLayer(withNeuronCount: 10, activationFunction: ReLUActivationFunction)
        neuralNetwork.addHiddenLayer(withNeuronCount: 10, activationFunction: softMaxActivationFunction)
    }

    func digitPredictions(withInputImage image: SampleImage) -> Matrix {
        return neuralNetwork.forwardPropagation(
            inputData: image.pixels.map { Double($0) / Double(SampleImage.Pixel.max) },
            outputLayerSize: 10
        )
    }
}
