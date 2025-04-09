const std = @import("std");
const CNN = @import("cnn.zig").CNN;
const Layer = @import("layers.zig");
const Activation = @import("functions_activate.zig");
const Loss = @import("functions_loss.zig");
const MNIST = @import("read_minst.zig");

const cnn = CNN(f64, 28, 28, Loss.MeanSquaredError, [_]Layer.LayerType{
  Layer.mergeAny([_]Layer.LayerType{
    Layer.mergeAny(.{
      [_]Layer.LayerType{
        Layer.getConvolver(4, 4, 1, 1, Activation.getPReLU(0.2)),
        Layer.getConvolver(4, 4, 2, 2, Activation.getPReLU(0.05)),
      },
      [_]Layer.LayerType{
        Layer.getConvolver(4, 4, 1, 1, Activation.getPReLU(0.2)),
        Layer.getConvolver(4, 4, 2, 2, Activation.getPReLU(0.05)),
      },
      [_]Layer.LayerType{
        Layer.getConvolver(4, 4, 1, 1, Activation.getPReLU(0.2)),
        Layer.getConvolver(4, 4, 2, 2, Activation.getPReLU(0.05)),
      },
      [_]Layer.LayerType{
        Layer.getConvolver(4, 4, 1, 1, Activation.getPReLU(0.2)),
        Layer.getConvolver(4, 4, 2, 2, Activation.getPReLU(0.05)),
      },
    }),
    Layer.getReshaper(26, 26),
    Layer.getConvolver(4, 4, 2, 2, Activation.getPReLU(0.1)),
    Layer.getConvolver(4, 4, 2, 2, Activation.getPReLU(0.1)),
  }),

  Layer.getFlattener(),
  Layer.getDense(7*7, Activation.getPReLU(0.1)),
  Layer.getDense(7*7, Activation.getPReLU(0.05)),
  Layer.getDense(10, Activation.NormalizeSquared),
});

const MNISTIterator = MNIST.GetMinstIterator(28, 28);

pub fn main() !void {
  @import("logger.zig").init();

  var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
  defer if (gpa.deinit() != .ok) @panic("Memory leak detected!!");
  const allocator = gpa.allocator();
  var rng = std.Random.DefaultPrng.init(@bitCast(std.time.timestamp()));

  var trainer: cnn.Trainer = undefined;
  var mnist_iterator = try MNISTIterator.init("./datasets/train-images.idx3-ubyte", "./datasets/train-labels.idx1-ubyte", allocator);
  defer mnist_iterator.free(allocator);

  trainer.reset(rng.random());
  for (0..4) |i| {
    trainer.train(mnist_iterator.randomIterator(rng.random(), mnist_iterator.count*2), .{
      .verbose = true,
      .batch_size = @intCast(i + 32),
      .learning_rate = @as(f64, 0.1) / @as(f64, @floatFromInt(1 + i)),
    });
  }

  var tester = trainer.toTester();

  const mnist_test_iterator = try MNISTIterator.init("./datasets/t10k-images.idx3-ubyte", "./datasets/t10k-labels.idx1-ubyte", allocator);
  defer mnist_test_iterator.free(allocator);

  const accuracy = tester.@"test"(mnist_test_iterator);
  std.debug.print("\n>>Final Accuracy: {d:.3}%\n", .{accuracy*100});
}

