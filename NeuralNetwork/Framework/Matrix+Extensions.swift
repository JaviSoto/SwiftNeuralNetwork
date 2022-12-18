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

    func rows(_ rows: some RandomAccessCollection<Int>) -> Matrix {
        precondition(rows.count > 0)
        precondition(rows.count <= self.rows)
        precondition(rows.allSatisfy { $0 >= 0 && $0 < self.rows })

        var matrix = Matrix(rows: rows.count, columns: columns, repeatedValue: 0)

        for (rowIndex, row) in rows.enumerated() {
            for column in 0..<columns {
                matrix[rowIndex, column] = self[row, column]
            }
        }

        precondition(matrix.rows == rows.count)
        precondition(matrix.columns == columns)

        return matrix
    }

    func columns(_ columns: some RandomAccessCollection<Int>) -> Matrix {
        precondition(columns.count > 0)
        precondition(columns.count <= self.columns)
        precondition(columns.allSatisfy { $0 >= 0 && $0 < self.columns })

        var matrix = Matrix(rows: rows, columns: columns.count, repeatedValue: 0)

        for row in 0..<rows {
            for (columnIndex, column) in columns.enumerated() {
                matrix[row, columnIndex] = self[row, column]
            }
        }

        precondition(matrix.rows == rows)
        precondition(matrix.columns == columns.count)

        return matrix
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
