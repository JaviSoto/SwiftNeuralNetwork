//
//  Helpers.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/15/22.
//

import Foundation

extension Int {
    var double: Double {
        get { return Double(self) }
        set { self = Int(exactly: newValue)! }
    }
}
