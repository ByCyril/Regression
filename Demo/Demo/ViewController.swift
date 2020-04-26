//
//  ViewController.swift
//  Demo
//
//  Created by Cyril Garcia on 4/23/20.
//  Copyright © 2020 Cyril Garcia. All rights reserved.
//

import Cocoa
import TensorFlow

class ViewController: NSViewController {
    
    @IBOutlet var epochField: NSTextField!
    @IBOutlet var learningRateField: NSTextField!
    
    var epochCount = 1000
    var learningRate: Float = 0.5
    
    var xVals: [Float] = []
    var yVals: [Float] = []
    
    var dots = [NSView]()
    var lineDots = [NSView]()
    
//    var model = LinearRegression()
    var model = PolynomialRegression()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.wantsLayer = true
    }
    
    override func mouseDown(with event: NSEvent) {
        let mousePosition = event.locationInWindow
        
        let dot = NSView(frame: CGRect(x: mousePosition.x, y: mousePosition.y, width: 5, height: 5))
        dot.wantsLayer = true
        
        dot.layer?.cornerRadius = dot.frame.height / 2
        dot.layer?.backgroundColor = NSColor.white.cgColor
        view.addSubview(dot)
        
        xVals.append(Float(mousePosition.x / 500.0))
        yVals.append(Float(mousePosition.y / 500.0))
        dots.append(dot)
    }
    
    @IBAction func train(_ sender: Any) {
        epochCount = (epochField.stringValue.isEmpty) ? 500 : Int(epochField.stringValue)!
        learningRate = (learningRateField.stringValue.isEmpty) ? 0.1 : Float(learningRateField.stringValue)!
        
        lineDots.forEach { (dot) in
            dot.layer?.backgroundColor = NSColor.clear.cgColor
            dot.removeFromSuperview()
        }
        lineDots.removeAll()
        
        train()
    }
    
    @IBAction func clear(_ sender: Any) {
        dots.append(contentsOf: lineDots)
        lineDots.removeAll()
        dots.forEach { (dot) in
            dot.layer?.backgroundColor = NSColor.clear.cgColor
            dot.removeFromSuperview()
        }
        dots.removeAll()
        xVals.removeAll()
        yVals.removeAll()
        model.clear()
    }
    
    func draw() {
        
        var x: Float = 0.0
        while x < 1.0 {
            x += 0.00125
            let y = CGFloat(model.callAsFunction(Tensor(x))[0].scalar!)
            
            let dot = NSView(frame: CGRect(x: CGFloat(x) * 500.0, y: y * 500.0, width: 5, height: 5))
            dot.wantsLayer = true
            
            dot.layer?.cornerRadius = dot.frame.height / 2
            dot.layer?.backgroundColor = NSColor.green.cgColor
            
            view.addSubview(dot)
            lineDots.append(dot)
        }
        
    }
    
    func train() {
        
        let inputs = Tensor(xVals)
        let outputs = Tensor(yVals)
        
        for _ in 1...epochCount {
            
            let optimizer = SGD(for: model, learningRate: learningRate)
            
            let (_, grads) = valueWithGradient(at: model) { model -> Tensor<Float> in
                return meanSquaredError(predicted: model.callAsFunction(inputs), expected: outputs)
            }
            optimizer.update(&model, along: grads)
        }
        
        draw()
    }
    
}
