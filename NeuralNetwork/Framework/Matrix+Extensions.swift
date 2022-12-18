//
//  Matrix+Extensions.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import Foundation
import SwiftMatrix
import Accelerate

extension Matrix {
    private struct MatrixMirror {
        let rows: Int
        let columns: Int
        var values: [Double]

        var matrix: Matrix {
            return Matrix(rows: rows, columns: columns, values: values)
        }

        init(from matrix: Matrix) {
            self = unsafeBitCast(matrix, to: MatrixMirror.self)
        }
    }

    func ReLU() -> Matrix {
        // A matrix where every element is at least 0
        return maxel(0, self)
    }

    func sum() -> Double {
        let sum = vDSP.sum(mutableValues)
        return sum.isFinite ? sum : Double.greatestFiniteMagnitude
    }

    var average: Double {
        var average: Double = 0
        let values = mutableValues
        let count = values.count

        for value in values {
            average += value / Double(count)
        }

        return average
    }

    func sumMatrix() -> Matrix {
        var result = Matrix(rows: 1, columns: columns, repeatedValue: 0)

        for row in 0..<rows {
            for column in 0..<columns {
                result[0, column] += self[row, column]
            }
        }

        return result
    }

    var mutableValues: [Double] {
        get {
            return MatrixMirror(from: self).values
        }
        set {
            newValue.forEach { $0.assertValid() }

            var mirror = MatrixMirror(from: self)
            mirror.values = newValue
            self = mirror.matrix
        }
    }

    func map(_ f: (Double) -> Double) -> Matrix {
        var copy = self
        copy.mutableValues = copy.mutableValues.map { value in
            let newValue = f(value)
            newValue.assertValid()
            return newValue
        }
        return copy
    }

    func row(_ rowIndex: Int) -> [Double] {
        var values: [Double] = []
        values.reserveCapacity(columns)

        for column in 0..<columns {
            values.append(self[rowIndex, column])
        }

        return values
    }

    func column(_ columnIndex: Int) -> [Double] {
        var values: [Double] = []
        values.reserveCapacity(rows)

        for row in 0..<rows {
            values.append(self[row, columnIndex])
        }

        return values
    }
}

extension Matrix {
    struct Shape: Equatable {
        let rows: Int
        let columns: Int

        postfix static func ′ (value: Shape) -> Shape {
            return .init(rows: value.columns, columns: value.rows)
        }
    }

    var shape: Shape {
        return .init(rows: rows, columns: columns)
    }
}

extension Matrix {
    func assertValid(file: StaticString = #file, line: UInt = #line) -> Matrix {
        #if DEBUG
        mutableValues.forEach { $0.assertValid(file: file, line: line) }
        #endif
        return self
    }
}
