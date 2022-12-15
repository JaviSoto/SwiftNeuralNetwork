//
//  NeuralNetwork.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/14/22.
//

import Foundation
import SwiftMatrix
import Accelerate

struct NeuralNetwork {
    typealias ActivationFunction = (Double) -> Double

    struct Layer {
        let weights: Matrix
        let biases: Matrix

        let neuronCount: Int

        var activationFunction: ActivationFunction

        init(previousLayerSize: Int, neurons: Int, activationFunction: @escaping ActivationFunction) {
            self.weights = Matrix.gaussianRandom(rows: previousLayerSize, columns: neurons)
            self.biases = Matrix.gaussianRandom(rows: neurons, columns: 1)
            self.activationFunction = activationFunction

            self.neuronCount = neurons
        }
    }

    private let inputLayerNeuronCount: Int

    private var layers: [Layer] = []

    init(inputLayerNeuronCount: Int) {
        self.inputLayerNeuronCount = inputLayerNeuronCount
    }

    mutating func addHiddenLayer(withNeuronCount neurons: Int, activationFunction: @escaping ActivationFunction) {
        layers.append(Layer(previousLayerSize: layers.last?.neuronCount ?? self.inputLayerNeuronCount, neurons: neurons, activationFunction: activationFunction))
    }

    func forwardPropagation(inputData: [Double], outputLayerSize: Int) -> Matrix {
        assert(inputData.count == inputLayerNeuronCount)

        var nextLayerInput = Matrix(rows: inputData.count, columns: 1, values: inputData)

        for layer in layers {
            let weightsApplied = layer.weights ° nextLayerInput + layer.biases
            let activations = weightsApplied.map(layer.activationFunction)
            nextLayerInput = activations
        }

        return nextLayerInput
    }

    // https://www.youtube.com/watch?v=w8yWXqWQYmU&t=1235s
//    func backwardPropagation(trainingData: Matrix, )
}

func ReLUActivationFunction(_ input: Matrix) -> Matrix {
    return input.ReLU()
}

func softMaxActivationFunction(_ input: Matrix) -> Matrix {
    let inputExp = exp(input)
    return inputExp / inputExp.sum()
}

extension Matrix {
    func ReLU() -> Matrix {
        // A matrix where every element is at least 0
        return maxel(0, self)
    }

    func sum() -> Double {
        return vDSP.sum(allElements())
    }

    func allElements() -> [Double] {
        // TODO: This is wasteful. Ideally the library exposes this
        var elements: [Double] = []
        elements.reserveCapacity(rows * columns)

        for row in 0..<rows {
            for column in 0..<columns {
                elements.append(self[row, column])
            }
        }

        return elements
    }

    func map(_ f: (Double) -> Double) -> Matrix {
        var matrix = self

        for row in 0..<matrix.rows {
            for column in 0..<matrix.columns {
                matrix[row, column] = f(matrix[row, column])
            }
        }

        return matrix
    }
}
