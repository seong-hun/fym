# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## September 9th, 2019

[0.2.2]

### Features

- **PID 모듈 추가**

	이제 PID 모듈이 추가됩니다. PID 모듈은 SISO로 구현되어 있으며 [관련 
	풀리퀘스트](https://github.com/fdcl-nrf/fym/pull/49)에 자세한 내용이 기재되어 
	있습니다.

- **[setup.py 추가](https://github.com/fdcl-nrf/fym/pull/74)**

	이제 `pip install -e .` 로 `fym` 모듈을 설치할 수 있습니다.


### Bug Fixes & Improvements

- **[선형화 모듈 개선](https://github.com/fdcl-nrf/fym/pull/82)**

	이전엔 jacobian 함수의 입력인자 순서가 고정되어 있었습니다. 이젠 추가적인 
	인자를 통해 어느 변수에 대해 jacobian을 구할 것인지 정할 수 있습니다.

- **[Plotting 모듈 그래프 한번에 뜨게 하도록 
	수정](https://github.com/fdcl-nrf/fym/pull/70)**
	- 이전엔 그래프가 하나씩 떠서 매번 닫아줘야 했습니다. 이젠 모든 그래프가 
		한번에 떠서 결과를 쉽게 확인할 수 있습니다.


## August 23th, 2019

[0.2.1]

- **Transfer to `fdcl-nrf` organization**

	The repository is transfered to the [`fdcl-nrf` 
	organization](https://github.com/fdcl-nrf/fym) with name `fym`.


## August 6th, 2019

[0.2.0]

### Features

- **Point-mass missile**

	Thus far, we only had an point-mass fixed-wing flight model and corresponding
	dynamic soaring environment. Now, a point-mass missile model and engagement
	environment is added to our `models` and `envs`.

### Bug Fixes & Improvements

- We added various type hints for the methods of `BaseEnv` and `BaseSystem`.

[Unreleased]: https://github.com/seong-hun/nrf-simulator/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/seong-hun/nrf-simulator/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/seong-hun/nrf-simulator/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/seong-hun/nrf-simulator/compare/v0.1.1...v0.2.0
