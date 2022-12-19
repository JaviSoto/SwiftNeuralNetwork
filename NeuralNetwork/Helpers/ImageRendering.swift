//
//  ImageRendering.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/14/22.
//

import Foundation
import SwiftUI

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
    public func asSwiftUIImage(width: UInt32) -> SwiftUI.Image {
        return self.map { $0.colorPixel }.asSwiftUIImage(width: width)
    }
}

extension [ColorPixel] {
    public func asSwiftUIImage(width: UInt32) -> SwiftUI.Image {
        let cgImage = asCGImage(width: width)

        return Image(decorative: cgImage, scale: 1)
    }

    #if os(macOS)
    public func asNSImage(width: UInt32) -> NSImage {
        let cgImage = asCGImage(width: width)

        return NSImage(cgImage: cgImage, size: .init(width: Double(width), height: Double(width)))
    }
    #endif

    func asCGImage(width: UInt32) -> CGImage {
        let width = Int(width)
        let pixels = self.map(\.sRGBAValue)
        precondition(pixels.count.isMultiple(of: width))

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
