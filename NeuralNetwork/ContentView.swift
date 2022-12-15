//
//  ContentView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/14/22.
//

import SwiftUI

struct ContentView: View {
    @State
    var trainingData: MNISTParser.DataSet?

    @State
    var randomItem: MNISTParser.DataSet.Item?

    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundColor(.accentColor)
            Text("Hello, world!")

            if let trainingData, let randomItem {
                SampleImageView(sampleImage: randomItem.image, width: trainingData.imageWidth)
                    .frame(width: 300)

                Text("\(randomItem.label.representedNumber)")
            }

            Button("New image") {
                updateImage()
            }
        }
        .padding()
        .onAppear {
            measure("Loading data") {
                self.trainingData = loadTrainingData()
                updateImage()
            }
        }
    }

    func updateImage() {
        randomItem = trainingData?.items.randomElement()!
    }
}

struct SampleImageView: View {
    let sampleImage: SampleImage
    let width: UInt32

    var body: some View {
        sampleImage.asSwiftUIImage(width: width)
            .resizable()
            .aspectRatio(1, contentMode: .fit)
    }
}

func measure<T>(_ name: String, _ f: () -> T) -> T {
    print("Starting '\(name)'")

    let start = CFAbsoluteTimeGetCurrent()
    let result = f()
    let end = CFAbsoluteTimeGetCurrent()

    let diff = end - start
    let duration = diff < 1 ? "\(Int(diff * 1000))ms" : "\(diff)s"

    print("'\(name)' took \(duration)")

    return result
}

private func loadTrainingData() -> MNISTParser.DataSet {
    let imageSetFile = Bundle.main.url(forResource: "train-images-idx3-ubyte", withExtension: nil)!
    let labelsFile = Bundle.main.url(forResource: "train-labels-idx1-ubyte", withExtension: nil)!

    return try! MNISTParser.loadData(imageSetFileURL: imageSetFile, labelDataFileURL: labelsFile)
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
