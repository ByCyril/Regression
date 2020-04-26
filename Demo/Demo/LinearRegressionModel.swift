//
//  LinearRegressionModel.swift
//  Demo
//
//  Created by Cyril Garcia on 4/25/20.
//  Copyright © 2020 Cyril Garcia. All rights reserved.
//

import Cocoa
import TensorFlow

struct LinearRegression: Layer {
    
    var m = Tensor([Float.random(in: 0...1)])
    var b = Tensor([Float.random(in: 0...1)])
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
//        y = mx + b
        return (m * input) + b
    }
    
    mutating func clear() {
        m = Tensor([Float.random(in: 0...1)])
        b = Tensor([Float.random(in: 0...1)])
    }
}
