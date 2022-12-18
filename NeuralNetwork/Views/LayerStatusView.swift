//
//  LayerStatusView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/18/22.
//

import SwiftUI
import SwiftMatrix

struct LayerStatusView: View {
    enum Visualization: CaseIterable {
        case weights
        case weightsApplied
        case activations

        var name: String {
            switch self {
            case .weights: return "Weights"
            case .weightsApplied: return "Weights Applied"
            case .activations: return "Activations"
            }
        }
    }

    private let layers: [LayerState]
    let visualization: Visualization

    init(layers: [NeuralNetwork.TrainingProgressObserver.LayerState], visualization: Visualization) {
        self.layers = layers.map { LayerState(layerState: $0, visualization: visualization) }
        self.visualization = visualization
    }

    fileprivate struct LayerState {
        let layerState: NeuralNetwork.TrainingProgressObserver.LayerState
        let weightRange: WeightRange

        struct WeightRange {
            let minValue: Double
            let maxValue: Double
        }

        init(layerState: NeuralNetwork.TrainingProgressObserver.LayerState, visualization: Visualization) {
            self.layerState = layerState

            self.weightRange = .init(
                minValue: min(layerState.matrixToVisualize(with: visualization)),
                maxValue: max(layerState.matrixToVisualize(with: visualization))
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

                    let matrixToVisualize = layer.item.layerState.matrixToVisualize(with: visualization)

                    ForEach((0..<layer.item.layerState.layer.neuronCount).indexed) { neuron in
                        let imageWidth: UInt32 = matrixToVisualize.columns > 10
                        ? UInt32(matrixToVisualize.columns.double.squareRoot())
                        : 1

                        NeuronView(
                            imageWidth: imageWidth,
                            neuronIndex: neuron.index,
                            neuronWeight: matrixToVisualize.rows([neuron.index]),
                            weightRange: layer.item.weightRange
                        )
                        .overlayingCoordinates()
                    }
                }
            }
        }
        .coordinateSpace(name: Self.coordinateSpaceName)
        .frame(maxWidth: .greatestFiniteMagnitude)
    }
}

extension View {
    func overlayingCoordinates() -> some View {
        return self.overlay(GeometryReader { proxy in
            let frame = proxy.frame(in: .named(LayerStatusView.coordinateSpaceName))

            Text("(\(Int(frame.origin.x)),\(Int(frame.origin.y)))")
                .frame(maxWidth: .greatestFiniteMagnitude)
                .lineLimit(0)

        })
    }
}

private extension NeuralNetwork.TrainingProgressObserver.LayerState {
    func matrixToVisualize(with visualization: LayerStatusView.Visualization) -> Matrix {
        switch visualization {
        case .weights:
            return layer.weights

            // These matrixes aren't square, so make them square to visualize them
        case .weightsApplied:
            return forwardPropagation?.weightsApplied.squareNumberOfColumns() ?? layer.weights
        case .activations:
            return forwardPropagation?.activationFunctionApplied.squareNumberOfColumns() ?? layer.weights
        }
    }
}

private extension Matrix {
    func squareNumberOfColumns() -> Matrix {
        let sizeSquareRoot = Int(columns.double.squareRoot().rounded(.down))
        return columns(0..<(sizeSquareRoot * sizeSquareRoot))
    }
}

private struct NeuronIndexPath: Hashable {
    let layerIndex: Int
    let neuronIndex: Int
}

//private struct NeuronViewPositionKey: PreferenceKey {
//    static var defaultValue: [NeuronIndexPath: CGSize] { [:] }
//    static func reduce(value: inout [UUID:CGSize], nextValue: () -> [UUID:CGSize]) {
//        let next = nextValue()
//        if let item = next.first {
//            value[item.key] = item.value
//        }
//    }
//}

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

        return column.map { value in
                if value < 0 {
                    let brightness = UInt8((minValue...0).projecting(clamped: value, into: 0...1) * 255)
                    return .init(red: brightness, green: 0, blue: 0)
                } else {
                    let brightness = UInt8((0...maxValue).projecting(clamped: value, into: 0...1) * 255)
                    return .init(red: 0, green: brightness, blue: 0)
                }
            }
    }
}

#if DEBUG

struct LayerStatusView_Previews: PreviewProvider {
    static var previews: some View {
        LayerStatusView(
            layers: [
                .init(layer: .init(previousLayerSize: 10, neurons: 10, activationFunction: .reLU), forwardPropagation: nil),
                .init(layer: .init(previousLayerSize: 10, neurons: 5, activationFunction: .reLU), forwardPropagation: nil),
            ],
            visualization: .weights
        )
    }
}

#endif
