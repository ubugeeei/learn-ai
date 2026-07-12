package learnai.tensor

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object TensorExecutionSuite extends TestSuite:
  override val name: String = "TensorExecution"

  override val tests: Vector[TestCase] = specify(
    test("right-aligned broadcasting maps matrix row and column operands") {
      val plan = Assert.right(BroadcastPlan.between(Shape(2, 1), Shape(1, 3)))
      Assert.equal(plan.output, Shape(2, 3))
      Assert.equal(
        plan.map(Vector(10.0, 20.0), Vector(1.0, 2.0, 3.0))(_ + _),
        Vector(11.0, 12.0, 13.0, 21.0, 22.0, 23.0)
      )
    },
    test("broadcast backward reduces replicated axes to original shapes") {
      val plan     = Assert.right(BroadcastPlan.between(Shape(2, 1), Shape(1, 3)))
      val upstream = Vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
      Assert.equal(plan.reduceGradient(upstream, forLeft = true), Vector(6.0, 15.0))
      Assert.equal(plan.reduceGradient(upstream, forLeft = false), Vector(5.0, 7.0, 9.0))
    },
    test("incompatible broadcast dimensions fail during planning") {
      val result = BroadcastPlan.between(Shape(2, 3), Shape(2, 4))
      Assert.isTrue(result.left.exists(_.contains("axis 1")))
    },
    test("transpose is a metadata view and materializes logical order") {
      val backing    = Vector(1, 2, 3, 4, 5, 6).map(_.toDouble)
      val original   = StridedView.contiguous(backing, Shape(2, 3))
      val transposed = original.transpose(0, 1)
      Assert.isTrue(transposed.backing eq original.backing)
      Assert.equal(transposed.shape, Shape(3, 2))
      Assert.equal(transposed.materialize, Vector(1, 4, 2, 5, 3, 6).map(_.toDouble))
    },
    test("batched matmul keeps batches isolated") {
      val left   = Tensor.constant(Shape(2, 1, 2), Vector(1, 2, 10, 20).map(_.toDouble))
      val right  = Tensor.constant(Shape(2, 2, 1), Vector(3, 4, 5, 6).map(_.toDouble))
      val output = BatchedMatmul(left, right)
      Assert.equal(output.shape, Shape(2, 1, 1))
      Assert.equal(output.values, Vector(11.0, 170.0))
    },
    test("graph lifetime accounts retained bytes and rejects double release") {
      val lifetime = GraphLifetime.retain(Vector(Shape(2, 3), Shape(4)), 4)
      Assert.equal(lifetime.bytesInUse, 40L)
      lifetime.release()
      Assert.equal(lifetime.bytesInUse, 0L)
      Assert.throws[IllegalArgumentException](lifetime.release())
      ()
    }
  )
