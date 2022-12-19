//
//  DigitDrawingView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/18/22.
//

import SwiftUI

struct DigitDrawingView: View {
    static let windowIdentifier = "es.javisoto.digit-drawing-view.window.id"

    static let size = CGSize(width: 400, height: 400)

    @ObservedObject
    private var appState = AppState.shared

    @State
    private var viewModel: NeuralNetworkViewModel?

    private let imageWidth: UInt32 = 28

    @State
    private var drawings: [Drawing] = []

    @State
    private var renderedDrawing: SampleImage?

    @State
    private var predictionOutcome: ImageRecognitionNeuralNetwork.PredictionOutcome = .init()

    @State
    private var tableOrder = [KeyPathComparator(\ImageRecognitionNeuralNetwork.PredictionOutcome.Digit.confidence, order: .reverse)]

    @State
    private var lineWidth: Double = 10

    var body: some View {
        HStack {
            VStack {
                GroupBox {
                    DrawingPad(
                        drawings: $drawings,
                        lineWidth: $lineWidth
                    )
                    .frame(width: Self.size.width, height: Self.size.height)

                    Button("Reset") {
                        drawings = []
                        predictionOutcome = .init()
                    }
                }
            }

            PredictionVisualizationView(
                attempt: renderedDrawing.map { .init(image: $0, expectedLabel: nil) },
                predictionOutcome: $predictionOutcome,
                tableOrder: $tableOrder
            )
        }
        .padding()
        .onChange(of: drawings) { drawings in
            updateImage()
        }
        .onChange(of: renderedDrawing) { image in
            if let image {
                updatePrediction(with: image)
            }
        }
        .onChange(of: viewModel?.state) { _ in
            if let renderedDrawing {
                updatePrediction(with: renderedDrawing)
            }
        }
        .onChange(of: appState) { appState in
            viewModel = appState.viewModel
        }
        .navigationSubtitle("Drawing")
    }

    @MainActor
    private func updateImage() {
        let view = DrawingView(
            drawings: drawings,
            lineWidth: lineWidth
        )
            .frame(width: Self.size.width, height: Self.size.height)

        let renderer = ImageRenderer(content: view)
        renderer.proposedSize = .init(width: Self.size.width, height: Self.size.height)
        renderer.isOpaque = true

        let renderedDrawing = SampleImage(renderer.cgImage!.scale(toWidth: Int(imageWidth)))
        self.renderedDrawing = renderedDrawing
    }

    private func updatePrediction(with image: SampleImage) {
        if let viewModel = appState.viewModel {
            self.predictionOutcome = viewModel.predictions(forImage: image)
            self.predictionOutcome.digits.sort(using: tableOrder)
        }
    }
}

private struct Drawing: Equatable {
    var points: [CGPoint] = []
}

private struct DrawingPad: View {
    @State
    private var currentDrawing = Drawing()

    @Binding
    var drawings: [Drawing]

    @Binding
    var lineWidth: Double

    var body: some View {
        GeometryReader { geometry in
            DrawingView(
                drawings: drawings + [currentDrawing],
                lineWidth: lineWidth
            )
            .gesture(
                DragGesture(minimumDistance: 0.1)
                    .onChanged { value in
                        let currentPoint = value.location
                        if geometry.frame(in: .local).contains(currentPoint) {
                            self.currentDrawing.points.append(currentPoint)
                        }
                    }
                    .onEnded { _ in
                        self.drawings.append(self.currentDrawing)
                        self.currentDrawing = Drawing()
                    }
            )
        }
        .aspectRatio(1, contentMode: .fill)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct DrawingView: View {
    var drawings: [Drawing]
    var lineWidth: Double

    var body: some View {
        Path { path in
            path.add(drawings)
        }
        .stroke(Color.white, lineWidth: self.lineWidth)
        .background(Color.black)
    }
}

private extension Path {
    mutating func add(_ drawings: [Drawing]) {
        for drawing in drawings {
            let points = drawing.points
            guard let firstPoint = points.first else { continue }

            self.move(to: firstPoint)

            for point in points.dropFirst() {
                self.addLine(to: point)
            }
        }
    }
}
