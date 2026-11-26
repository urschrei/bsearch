//! Text recognition in images, with the Vision framework.

use anyhow::Result;
use objc2::AnyThread;
use objc2::rc::autoreleasepool;
use objc2_foundation::NSArray;
use objc2_foundation::NSData;
use objc2_foundation::NSDictionary;
use objc2_vision::VNImageRequestHandler;
use objc2_vision::VNRecognizeTextRequest;
use objc2_vision::VNRequest;
use objc2_vision::VNRequestTextRecognitionLevel;

/// Recognise the text in an encoded image, in any format Core Image reads.
///
/// Returns one line per text region that Vision finds, in the order Vision
/// gives them, joined by newlines. An image without text gives an empty
/// string. The call blocks until recognition is complete; the first call on
/// a machine can take tens of seconds while Vision prepares its models.
pub fn recognise_text(image: &[u8]) -> Result<String> {
    autoreleasepool(|_| {
        let data = NSData::with_bytes(image);
        let handler = VNImageRequestHandler::initWithData_options(
            VNImageRequestHandler::alloc(),
            &data,
            &NSDictionary::new(),
        );

        let request = VNRecognizeTextRequest::new();
        request.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);
        request.setUsesLanguageCorrection(true);
        request.setAutomaticallyDetectsLanguage(true);

        let as_request: &VNRequest = &request;
        handler
            .performRequests_error(&NSArray::from_slice(&[as_request]))
            .map_err(|e| anyhow::anyhow!("Vision could not read the image: {e}"))?;

        let lines: Vec<String> = request
            .results()
            .iter()
            .flat_map(|observations| observations.iter())
            .filter_map(|observation| observation.topCandidates(1).firstObject())
            .map(|candidate| candidate.string().to_string())
            .filter(|line| !line.trim().is_empty())
            .collect();
        Ok(lines.join("\n"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_is_recognised_line_by_line() {
        let image = include_bytes!("../testdata/sign.jpg");
        let text = recognise_text(image).expect("recognition failed");
        assert_eq!(
            text,
            "Ferrets steal socks\nQuarterly report 2026\nOpening hours 9 to 5"
        );
    }

    #[test]
    fn test_bytes_that_are_not_an_image_are_an_error() {
        assert!(recognise_text(b"<html>not an image</html>").is_err());
    }
}
