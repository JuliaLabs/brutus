module NVVM
  function barrier0()
    Base.inferencebarrier(nothing)::Nothing
  end
end