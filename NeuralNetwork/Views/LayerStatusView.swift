//
//  LayerStatusView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/18/22.
//

import SwiftUI
import SwiftMatrix

struct LayerStatusView: View {
    private let layers: [LayerState]

    init(layers: [NeuralNetwork.TrainingProgressObserver.LayerState]) {
        self.layers = layers.map(LayerState.init)
    }

    fileprivate struct LayerState {
        let layerState: NeuralNetwork.TrainingProgressObserver.LayerState
        let weightRange: WeightRange

        struct WeightRange {
            let minValue: Double
            let maxValue: Double
        }

        init(layerState: NeuralNetwork.TrainingProgressObserver.LayerState) {
            self.layerState = layerState
            self.weightRange = .init(
                minValue: min(layerState.layer.weights),
                maxValue: max(layerState.layer.weights)
            )
        }
    }

    fileprivate static let coordinateSpaceName = "LayerStatusView"

    var body: some View {
        HStack(spacing: 100) {
            ForEach(layers.indexed) { layer in
                VStack {
                    let isLastLayer = layer.index == layers.count - 1
                    Text(isLastLayer ? "Output Layer" : "Layer \(layer.index + 1)")
                        .bold()

                    ForEach((0..<layer.item.layerState.layer.neuronCount).indexed) { neuron in
                        let imageWidth: UInt32 = layer.item.layerState.layer.weights.columns > 10
                        ? UInt32(layer.item.layerState.layer.weights.columns.double.squareRoot())
                        : 1

                        NeuronView(
                            imageWidth: imageWidth,
                            neuronIndex: neuron.index,
                            neuronWeight: layer.item.layerState.layer.weights.rows([neuron.index]),
                            weightRange: layer.item.weightRange
                        )
                    }
                }
            }
        }
        .coordinateSpace(name: Self.coordinateSpaceName)
        .frame(maxWidth: .greatestFiniteMagnitude)
    }
}

private struct NeuronView: View {
    let imageWidth: UInt32
    let neuronIndex: Int
    let neuronWeight: Matrix
    let weightRange: LayerStatusView.LayerState.WeightRange

    let pixels: [ColorPixel]

    init(imageWidth: UInt32, neuronIndex: Int, neuronWeight: Matrix, weightRange: LayerStatusView.LayerState.WeightRange) {
        assert(neuronWeight.rows == 1)

        self.imageWidth = imageWidth
        self.neuronIndex = neuronIndex
        self.neuronWeight = neuronWeight
        self.weightRange = weightRange
        self.pixels = neuronWeight.pixels(minValue: weightRange.minValue, maxValue: weightRange.maxValue)
    }

    var body: some View {
        ZStack {
            PixelArrayImageView(pixels: pixels, width: imageWidth, lazy: false)
                .frame(height: 100)

            Text("\(neuronIndex)")
                .blendMode(.colorDodge)
        }
        .clipShape(RoundedRectangle(cornerSize: CGSize(width: 5, height: 5)))
    }
}

private extension Matrix {
    func pixels(minValue: Double, maxValue: Double) -> [ColorPixel] {
        assert(rows == 1)

        let column = self.row(0)
        let range = (minValue...maxValue)

        return column.lazy
            .map { range.projecting(clamped: $0, into: -1...1) }
            .map { value in
                let brightness = UInt8(abs(value) * 255)
                if value < 0 {
                    return .init(red: brightness, green: 0, blue: 0)
                } else {
                    return .init(red: 0, green: brightness, blue: 0)
                }
            }
    }
}

struct LayerStatusView_Previews: PreviewProvider {
    static var previews: some View {
        LayerStatusView(
            layers: [
                .init(layer: .init(previousLayerSize: 10, neurons: 10, activationFunction: .reLU), forwardPropagation: nil),
                .init(layer: .init(previousLayerSize: 10, neurons: 5, activationFunction: .reLU), forwardPropagation: nil),
            ]
        )
    }
}
