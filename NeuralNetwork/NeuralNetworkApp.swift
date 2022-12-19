//
//  NeuralNetworkApp.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/14/22.
//

import SwiftUI

final class AppState: ObservableObject, Equatable {
    static let shared = AppState()

    @Published
    var viewModel: NeuralNetworkViewModel?

    static func == (lhs: AppState, rhs: AppState) -> Bool {
        return lhs.viewModel === rhs.viewModel
    }
}

@main
struct NeuralNetworkApp: App {
    @StateObject
    var appState = AppState.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
        }

        WindowGroup(id: DigitDrawingView.windowIdentifier) {
            DigitDrawingView()
        }
    }
}
