//
//  ImageRendering.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/14/22.
//

import Foundation
import SwiftUI

extension SampleImage {
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

    private func asCGImage(width: UInt32) -> CGImage {
        let width = Int(width)
        let pixels = pixels.map(\.sRGBAValue)
        precondition(pixels.count.isMultiple(of: width))

        let providerRef = CGDataProvider(data: NSData(bytes: pixels, length: pixels.count * 4))

        return CGImage(
            width: width,
            height: pixels.count / width,
            bitsPerComponent: 8,
            bitsPerPixel: 4 * 8,
            bytesPerRow: 4 * width,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: .byteOrder32Little,
            provider: providerRef!,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )!
    }
}

extension SampleImage.Pixel {
    var sRGBAValue: UInt32 {
        let color = UInt32(self)
        return color << 24 + color << 16 + color << 8 + 255
    }
}

#if DEBUG
extension [Double] {
    func asNSImage(width: UInt32) -> NSImage {
        return SampleImage(pixels: self.map { UInt8($0 * 255) })
            .asNSImage(width: width)
    }
}
#endif
