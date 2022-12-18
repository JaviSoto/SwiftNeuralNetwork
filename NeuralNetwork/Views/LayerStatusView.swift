//
//  LayerStatusView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/18/22.
//

import SwiftUI
import SwiftMatrix
import Accelerate

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

    private let layers: [NeuralNetwork.TrainingProgressObserver.LayerState]
    let visualization: Visualization

    @State
    private var neuronFrames: [NeuronIndexPath: CGRect] = [:]

    init(layers: [NeuralNetwork.TrainingProgressObserver.LayerState], visualization: Visualization) {
        self.layers = layers
        self.visualization = visualization
    }

    fileprivate static let coordinateSpaceName = "LayerStatusView"

    var body: some View {
        ZStack {
            HStack {
                ForEach(layers.indexed) { layer in
                    Spacer()
                    VStack {
                        let isLastLayer = layer.index == layers.count - 1
                        Text(isLastLayer ? "Output Layer" : "Layer \(layer.index + 1)")
                            .bold()

                        let matrixToVisualize = layer.item.matrixToVisualize(with: visualization)

                        ForEach((0..<layer.item.layer.neuronCount).indexed) { neuron in
                            let imageWidth: UInt32 = matrixToVisualize.columns > 10
                            ? UInt32(matrixToVisualize.columns.double.squareRoot())
                            : 1

                            NeuronView(
                                imageWidth: imageWidth,
                                neuronIndexPath: .init(layerIndex: layer.index, neuronIndex: neuron.index),
                                neuronMatrix: matrixToVisualize.rows([neuron.index])
                            )
                        }
                    }
                    Spacer()
                }
            }

            BiasLines(
                layers: layers,
                neuronFrames: neuronFrames
            )
        }
        .coordinateSpace(name: Self.coordinateSpaceName)
        .onPreferenceChange(NeuronViewFrameKey.self) { neuronFrames in
            self.neuronFrames = neuronFrames
        }
        .frame(maxWidth: .greatestFiniteMagnitude)
    }
}

private struct BiasLines: View {
    let layers: [NeuralNetwork.TrainingProgressObserver.LayerState]
    let neuronFrames: [NeuronIndexPath: CGRect]

    var body: some View {
        if layers.isEmpty {
            EmptyView()
        } else {
            let biasRange = vDSP.minimum(layers.map { min($0.layer.biases) })...vDSP.maximum(layers.map { max($0.layer.biases) })

            ForEach(layers.dropLast().indexed) { layer in
                ForEach((0..<layer.item.layer.neuronCount).indexed) { neuron in
                    let startIndexPath = NeuronIndexPath(layerIndex: layer.index, neuronIndex: neuron.index)
                    if let start = neuronFrames[startIndexPath] {
                        // Start points to the (1, 0.5) point
                        let start = start.offsetBy(
                            dx: start.size.width,
                            dy: start.size.height / 2
                        )
                        ForEach((0..<layers[layer.index + 1].layer.neuronCount).indexed) { nextLayerNeuron in
                            let endIndexPath = NeuronIndexPath(layerIndex: layer.index + 1, neuronIndex: nextLayerNeuron.index)
                            if let end = neuronFrames[endIndexPath] {
                                BiasLine(
                                    bias: layers[layer.index].layer.biases.row(neuron.index)[0],
                                    biasRange: biasRange,
                                    neuronIndex: neuron.index,
                                    neuronCount: layer.item.layer.neuronCount,
                                    startFrame: start,
                                    endFrame: end
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

private struct BiasLine: View {
    let bias: Double
    let biasRange: ClosedRange<Double>
    let neuronIndex: Int
    let neuronCount: Int
    let startFrame: CGRect
    let endFrame: CGRect

    var body: some View {
        let relativeBias: Double = bias > 0
        ? (0...biasRange.upperBound).projecting(clamped: bias, into: 0...1)
        : (biasRange.lowerBound...0).projecting(clamped: bias, into: 0...1)

        let lineWidth: Double = relativeBias * 5
        let lineColor: Color = bias > 0
        ? Color.blue.opacity(abs(relativeBias))
        : Color.red.opacity(abs(relativeBias))

        Path { path in
            // End points to the (0, 0.5) point
            let offsetEnd = endFrame.offsetBy(
                dx: 0,
                dy: endFrame.size.height / 2
            )
            let curveControlPoint: CGRect = startFrame.offsetBy(
                dx: 100,
                dy: Double((Double(neuronIndex) - Double(neuronCount / 2)) * 25)
            )
            path.move(to: startFrame.origin)
            path.addQuadCurve(to: offsetEnd.origin, control: curveControlPoint.origin)
        }
        .strokedPath(.init(lineWidth: lineWidth))
        .foregroundColor(lineColor)
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

private struct NeuronViewFrameKey: PreferenceKey {
    static var defaultValue: [NeuronIndexPath: CGRect] { [:] }

    static func reduce(value: inout [NeuronIndexPath: CGRect], nextValue: () -> [NeuronIndexPath: CGRect]) {
        for (key, newValue) in nextValue() {
            value[key] = newValue
        }
    }
}

private struct NeuronView: View {
    let indexPath: NeuronIndexPath
    let imageWidth: UInt32
    let neuronMatrix: Matrix

    let pixels: [ColorPixel]

    init(imageWidth: UInt32, neuronIndexPath: NeuronIndexPath, neuronMatrix: Matrix) {
        assert(neuronMatrix.rows == 1)

        self.imageWidth = imageWidth
        self.indexPath = neuronIndexPath
        self.neuronMatrix = neuronMatrix
        self.pixels = neuronMatrix.pixels(minValue: min(neuronMatrix), maxValue: max(neuronMatrix))
    }

    var body: some View {
        ZStack {
            PixelArrayImageView(pixels: pixels, width: imageWidth, lazy: false)
                .frame(height: 100)

            Text("\(indexPath.neuronIndex)")
                .padding()
                .foregroundColor(.white)
                .blendMode(.plusLighter)
        }
        .clipShape(RoundedRectangle(cornerSize: CGSize(width: 5, height: 5)))
        .overlay(GeometryReader { proxy in
            let frame = proxy.frame(in: .named(LayerStatusView.coordinateSpaceName))

            Rectangle()
                .hidden()
                .preference(key: NeuronViewFrameKey.self, value: [indexPath: frame])
        })
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
