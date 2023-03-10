//
//  ContentView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/14/22.
//

import SwiftUI
import SwiftMatrix

struct ContentView: View {
    @State
    private var trainingData: MNISTData?

    var body: some View {
        Group {
            if let trainingData {
                NeuralNetworkView(trainingData: trainingData)
            } else {
                VStack {
                    Text("Loading training data")
                    ProgressView()
                }
            }
        }.task {
#if DEBUG
            let maxCount = 1000
#else
            let maxCount: Int? = nil
#endif

            trainingData = await DataLoading.loadTrainingData(maxCount: maxCount)
        }
    }
}
