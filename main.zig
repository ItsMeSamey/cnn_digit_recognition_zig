const std = @import("std");
const CNN = @import("cnn.zig").CNN;
const Layer = @import("layers.zig");
const Activation = @import("functions_activate.zig");
const Loss = @import("functions_loss.zig");
const MNIST = @import("read_minst.zig");

const cnn = CNN(f64, 28, 28, Loss.MeanSquaredError, [_]Layer.LayerType{

  // Layer.mergeAny([_]Layer.LayerType{
  //   Layer.mergeAny(.{
  //     [_]Layer.LayerType{
  //       Layer.getConvolver(4, 4, 4, 4, Activation.getPReLU(0.05)),
  //     },
  //     [_]Layer.LayerType{
  //       Layer.getConvolver(4, 4, 4, 4, Activation.getPReLU(0.05)),
  //     },
  //     [_]Layer.LayerType{
  //       Layer.getConvolver(4, 4, 4, 4, Activation.getPReLU(0.05)),
  //     },
  //     [_]Layer.LayerType{
  //       Layer.getConvolver(4, 4, 4, 4, Activation.getPReLU(0.05)),
  //     },
  //   }),
  // }),

  Layer.getFlattener(),
  // Layer.getDense(28*28, Activation.getPReLU(0.5)),
  // Layer.getDense(28*28, Activation.getPReLU(0.125)),
  // Layer.getDense(14*14, Activation.getPReLU(0.125)),
  // Layer.getDense(14*14, Activation.getPReLU(0.125)),
  Layer.getDense(7*7, Activation.getPReLU(0.125)),
  Layer.getDense(7*7, Activation.getPReLU(0.125)),
  Layer.getDense(10, Activation.NormalizeSquared),
});

const MNISTIterator = MNIST.GetMinstIterator(28, 28);

pub fn main() !void {
  @import("logger.zig").init();

  var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
  defer if (gpa.deinit() != .ok) @panic("Memory leak detected!!");
  const allocator = gpa.allocator();

  var rng = std.Random.DefaultPrng.init(1);
  var trainer: cnn.Trainer = undefined;
  trainer.reset(rng.random());

  var mnist_iterator = try MNISTIterator.init("./datasets/train-images.idx3-ubyte", "./datasets/train-labels.idx1-ubyte", allocator);
  defer mnist_iterator.free(allocator);

  inline for (0..8) |i| {
    trainer.train(mnist_iterator.randomIterator(rng.random(), @intCast(mnist_iterator.count*(i+1))), .{
      .verbose = true,
      .batch_size = @intCast(mnist_iterator.count/(10*(i+1)*(i+1))),
      .learning_rate = 100 / @exp(@as(f64, @floatFromInt(i+1))),
    });
  }
  // inline for (0..2) |i| {
  //   trainer.train(mnist_iterator.randomIterator(rng.random(), mnist_iterator.count), .{
  //     .verbose = true,
  //     .batch_size = @intCast((i+1) * 16),
  //     .learning_rate = @as(f64, 0.01) / @as(f64, @floatFromInt(1+i)),
  //   });
  // }

  var tester = trainer.toTester();

  var mnist_test_iterator = try MNISTIterator.init("./datasets/t10k-images.idx3-ubyte", "./datasets/t10k-labels.idx1-ubyte", allocator);
  defer mnist_test_iterator.free(allocator);

  const accuracy = tester.@"test"(mnist_test_iterator, true);
  std.debug.print("\n>>Final Accuracy(testerr): {d:.3}%\n", .{accuracy*100});
}

