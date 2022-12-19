//
//  ImageRendering.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/14/22.
//

import Foundation
import SwiftUI
import SwiftMatrix
import Accelerate

struct ColorPixel {
    let red: UInt8
    let green: UInt8
    let blue: UInt8
}

extension SampleImage.Pixel {
    var colorPixel: ColorPixel {
        return .init(red: self, green: self, blue: self)
    }
}

extension [SampleImage.Pixel] {
    func asSwiftUIImage(width: UInt32) -> SwiftUI.Image {
        return self.map { $0.colorPixel }.asSwiftUIImage(width: width)
    }

    func asCGImage(width: UInt32, height: UInt32) -> CGImage {
        return self.map { $0.colorPixel }.asCGImage(width: width, height: height)
    }
}

extension [ColorPixel] {
    func asSwiftUIImage(width: UInt32) -> SwiftUI.Image {
        let cgImage = asCGImage(width: width, height: width)

        return Image(decorative: cgImage, scale: 1)
    }

    #if os(macOS)
    func asNSImage(width: UInt32, height: UInt32) -> NSImage {
        let cgImage = asCGImage(width: width, height: height)

        return NSImage(cgImage: cgImage, size: .init(width: Double(width), height: Double(height)))
    }
    #endif

    func asCGImage(width: UInt32, height: UInt32) -> CGImage {
        let width = Int(width)
        let pixels = self.map(\.sRGBAValue)
        precondition(pixels.count.isMultiple(of: width), "Expected pixels to be multiple of \(width), but is \(pixels.count) instead")

        let providerRef = CGDataProvider(data: NSData(bytes: pixels, length: pixels.count * 4))

        return CGImage(
            width: width,
            height: pixels.count / width,
            bitsPerComponent: 8,
            bitsPerPixel: 4 * 8,
            bytesPerRow: 4 * width,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue).union(.byteOrder32Little),
            provider: providerRef!,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )!
    }
}

extension SampleImage {
    init(_ cgImage: CGImage) {
        let bitsPerComponent = cgImage.bitsPerComponent
        let bytesPerRow = cgImage.width * bitsPerComponent / 8
        let totalBytes = cgImage.height * bytesPerRow

        var intensities = [SampleImage.Pixel](repeating: 0, count: totalBytes)

        let context = CGContext(
            data: &intensities,
            width: cgImage.width,
            height: cgImage.height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue).rawValue
        )!

        context.draw(cgImage, in: CGRect(
            x: 0,
            y: 0,
            width: Int(cgImage.width),
            height: Int(cgImage.height)
        ))

        self = .init(pixels: intensities, width: UInt32(cgImage.width), height: UInt32(cgImage.height))
    }
}

extension CGImage {
    func scale(toWidth width: Int) -> CGImage {
        let context = CGContext(
            data: nil,
            width: Int(width),
            height: Int(width),
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: 0,
            space: colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: bitmapInfo.rawValue
        )!
        context.interpolationQuality = .high
        context.draw(self, in: CGRect(origin: .zero, size: CGSize(width: width, height: width)))

        return context.makeImage()!
    }
}

private extension ColorPixel {
    var sRGBAValue: UInt32 {
        return UInt32(red) << 24 + UInt32(green) << 16 + UInt32(blue) << 8 + 255
    }
}

// MARK: - Image Manipulation

private extension Matrix {
    init(_ sampleImage: SampleImage) {
        self.init(
            rows: Int(sampleImage.height),
            columns: Int(sampleImage.width),
            values: sampleImage.pixels.map { Double($0) / Double(SampleImage.Pixel.max) }
        )
    }
}

private extension SampleImage {
    init(_ matrix: Matrix) {
        let pixels = matrix.mutableValues.map { SampleImage.Pixel($0 * 255) }
        self = SampleImage(pixels: pixels, width: UInt32(matrix.columns), height: UInt32(matrix.rows))
    }
}

extension SampleImage {
    /// Returns a version of `self` that has no "blank" space outside of the contents.
    func croppingContents() -> SampleImage {
        let pixelMatrix = Matrix(self)

        var firstRowWithContent: Int?
        var firstColumnWithContent: Int?
        var lastRowWithContent: Int?
        var lastColumnWithContent: Int?

        for row in 0..<pixelMatrix.rows {
            for column in 0..<pixelMatrix.columns {
                guard pixelMatrix[row, column] > 0 else { continue }

                if column < (firstColumnWithContent ?? Int.max) {
                    firstColumnWithContent = column
                }

                if row < (firstRowWithContent ?? Int.max) {
                    firstRowWithContent = row
                }

                if column > (lastColumnWithContent ?? Int.min) {
                    lastColumnWithContent = column
                }

                if row > (lastRowWithContent ?? Int.min) {
                    lastRowWithContent = row
                }
            }
        }

        precondition(firstRowWithContent != nil)
        precondition(firstColumnWithContent != nil)
        precondition(lastRowWithContent != nil)
        precondition(lastColumnWithContent != nil)

        let rows = Array(firstRowWithContent!...lastRowWithContent!)
        let columns = Array(firstColumnWithContent!...lastColumnWithContent!)

        var croppedMatrix = Matrix(rows: rows.count, columns: columns.count, repeatedValue: 0)

        for (newRowIndex, originalRowIndex) in rows.enumerated() {
            for (newColumnIndex, originalColumnIndex) in columns.enumerated() {
                croppedMatrix[newRowIndex, newColumnIndex] = pixelMatrix[originalRowIndex, originalColumnIndex]
            }
        }

        return SampleImage(croppedMatrix)
    }

    func addingBlankSpace(toCenterInWidth width: UInt32, height: UInt32, withXOffset xOffset: Int, yOffset: Int) -> SampleImage {
        let pixelMatrix = Matrix(self)

        let rowsToAdd = Int(width) - pixelMatrix.rows
        let columnsToAdd = Int(height) - pixelMatrix.columns

        precondition(rowsToAdd >= 0)
        precondition(columnsToAdd >= 0)

        let topPadding = max(-1, min(rowsToAdd / 2 + yOffset, Int(height) - pixelMatrix.rows - 1))
        let leftPadding = max(-1, min(columnsToAdd / 2 + xOffset, Int(width) - pixelMatrix.columns - 1))

        var newMatrix = Matrix(rows: Int(height), columns: Int(width), repeatedValue: 0)

        for (originalRowIndex, newRowIndex) in ((topPadding + 1)...(pixelMatrix.rows + topPadding)).enumerated() {
            for (originalColumnIndex, newColumnIndex) in ((leftPadding + 1)...(pixelMatrix.columns + leftPadding)).enumerated() {
                newMatrix[newRowIndex, newColumnIndex] = pixelMatrix[originalRowIndex, originalColumnIndex]
            }
        }

        return SampleImage(newMatrix)
    }

    func scaling(byFactor scaleFactor: Double) -> SampleImage {
        guard scaleFactor != 1 else { return self }

        let newWidth = UInt32(Double(width) * scaleFactor)
        let newHeight = UInt32(Double(height) * scaleFactor)

        return self.scaling(toWidth: newWidth, height: newHeight)
    }

    func scaling(toWidth newWidth: UInt32, height newHeight: UInt32) -> SampleImage {
        var pixels = pixels
        var imageBuffer: vImage_Buffer!
        var destinationBuffer: vImage_Buffer!

        return pixels.withUnsafeMutableBufferPointer { pointer -> SampleImage in
            imageBuffer = vImage_Buffer(data: pointer.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: Int(width))
            imageBuffer.rowBytes = Int(width)
            destinationBuffer = try! vImage_Buffer(width: Int(newWidth), height: Int(newHeight), bitsPerPixel: 8)
            destinationBuffer.rowBytes = Int(newWidth)
            let error = vImageScale_Planar8(&imageBuffer, &destinationBuffer, nil, vImage_Flags(kvImageNoFlags))

            precondition(error == kvImageNoError)

            defer {
                destinationBuffer.free()
            }

            let scaledPixels = [Pixel](destinationBuffer)

            return SampleImage(pixels: scaledPixels, width: newWidth, height: newHeight)
        }
    }

    func rotating(by angle: Angle) -> SampleImage {
        var pixels = pixels

        return pixels.withUnsafeMutableBufferPointer { pointer -> SampleImage in
            var imageBuffer = vImage_Buffer(data: pointer.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: Int(width))
            let error = vImageRotate_Planar8(&imageBuffer, &imageBuffer, nil, Float(angle.radians), Pixel_8(0), vImage_Flags(kvImageBackgroundColorFill))

            precondition(error == kvImageNoError)

            let pixels = [Pixel](imageBuffer)

            return SampleImage(pixels: pixels, width: width, height: height)
        }
    }

    func randomlyShiftingContents() -> SampleImage {
        let maxRotationDegrees: Double = 30

        let result = self
            .rotating(by: .degrees(Double.random(in: -maxRotationDegrees...maxRotationDegrees)))

        return result
    }
}

private extension [SampleImage.Pixel] {
    init(_ accelerateBuffer: vImage_Buffer) {
        let bufferData = accelerateBuffer.data!
        let length = Int(accelerateBuffer.width * accelerateBuffer.height)

        let pixelPointer = bufferData.bindMemory(to: SampleImage.Pixel.self, capacity: length)
        let pixelBuffer = UnsafeBufferPointer(start: pixelPointer, count: length)

        self = Array(pixelBuffer)
    }
}

private extension Matrix {
    var asNSImage: NSImage {
        return SampleImage(self).pixels.map(\.colorPixel).asNSImage(width: UInt32(self.rows), height: UInt32(self.columns))
    }
}
