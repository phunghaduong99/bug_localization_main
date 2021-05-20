/*******************************************************************************
 * Copyright (c) 2010 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 ******************************************************************************/

package org.eclipse.e4.ui.tests.reconciler;

import java.util.Collection;
import org.eclipse.e4.ui.model.application.MApplication;
import org.eclipse.e4.ui.model.application.ui.advanced.MPerspective;
import org.eclipse.e4.ui.model.application.ui.advanced.MPerspectiveStack;
import org.eclipse.e4.ui.model.application.ui.advanced.impl.AdvancedFactoryImpl;
import org.eclipse.e4.ui.model.application.ui.basic.MWindow;
import org.eclipse.e4.ui.model.application.ui.basic.impl.BasicFactoryImpl;
import org.eclipse.e4.ui.workbench.modeling.ModelDelta;
import org.eclipse.e4.ui.workbench.modeling.ModelReconciler;

public abstract class ModelReconcilerPerspectiveTest extends
		ModelReconcilerTest {

	public void testPerspective_Windows_Add() {
		MApplication application = createApplication();
		MWindow window = createWindow(application);

		MPerspectiveStack perspectiveStack = AdvancedFactoryImpl.eINSTANCE
				.createPerspectiveStack();
		window.getChildren().add(perspectiveStack);

		MPerspective perspective = AdvancedFactoryImpl.eINSTANCE
				.createPerspective();
		perspectiveStack.getChildren().add(perspective);

		saveModel();

		ModelReconciler reconciler = createModelReconciler();
		reconciler.recordChanges(application);

		MWindow nestedWindow = BasicFactoryImpl.eINSTANCE.createWindow();
		nestedWindow.setElementId("nested");
		perspective.getWindows().add(nestedWindow);

		Object state = reconciler.serialize();

		application = createApplication();
		window = application.getChildren().get(0);
		perspectiveStack = (MPerspectiveStack) window.getChildren().get(0);
		perspective = perspectiveStack.getChildren().get(0);

		Collection<ModelDelta> deltas = constructDeltas(application, state);

		assertEquals(1, application.getChildren().size());
		assertEquals(window, application.getChildren().get(0));
		assertEquals(1, window.getChildren().size());
		assertEquals(perspectiveStack, window.getChildren().get(0));
		assertEquals(1, perspectiveStack.getChildren().size());
		assertEquals(perspective, perspectiveStack.getChildren().get(0));
		assertEquals(0, perspective.getWindows().size());

		applyAll(deltas);

		assertEquals(1, application.getChildren().size());
		assertEquals(window, application.getChildren().get(0));
		assertEquals(1, window.getChildren().size());
		assertEquals(perspectiveStack, window.getChildren().get(0));
		assertEquals(1, perspectiveStack.getChildren().size());
		assertEquals(perspective, perspectiveStack.getChildren().get(0));
		assertEquals(1, perspective.getWindows().size());

		nestedWindow = (MWindow) perspective.getWindows().get(0);
		assertEquals("nested", nestedWindow.getElementId());
	}

	public void testPerspective_Windows_Remove() {
		MApplication application = createApplication();
		MWindow window = createWindow(application);

		MPerspectiveStack perspectiveStack = AdvancedFactoryImpl.eINSTANCE
				.createPerspectiveStack();
		window.getChildren().add(perspectiveStack);

		MPerspective perspective = AdvancedFactoryImpl.eINSTANCE
				.createPerspective();
		perspectiveStack.getChildren().add(perspective);

		MWindow nestedWindow = BasicFactoryImpl.eINSTANCE.createWindow();
		nestedWindow.setElementId("nested");
		perspective.getWindows().add(nestedWindow);

		saveModel();

		ModelReconciler reconciler = createModelReconciler();
		reconciler.recordChanges(application);

		perspective.getWindows().remove(0);

		Object state = reconciler.serialize();

		application = createApplication();
		window = application.getChildren().get(0);
		perspectiveStack = (MPerspectiveStack) window.getChildren().get(0);
		perspective = perspectiveStack.getChildren().get(0);
		nestedWindow = perspective.getWindows().get(0);

		Collection<ModelDelta> deltas = constructDeltas(application, state);

		assertEquals(1, application.getChildren().size());
		assertEquals(window, application.getChildren().get(0));
		assertEquals(1, window.getChildren().size());
		assertEquals(perspectiveStack, window.getChildren().get(0));
		assertEquals(1, perspectiveStack.getChildren().size());
		assertEquals(perspective, perspectiveStack.getChildren().get(0));
		assertEquals(1, perspective.getWindows().size());
		assertEquals(nestedWindow, perspective.getWindows().get(0));
		assertEquals("nested", nestedWindow.getElementId());

		applyAll(deltas);

		assertEquals(1, application.getChildren().size());
		assertEquals(window, application.getChildren().get(0));
		assertEquals(1, window.getChildren().size());
		assertEquals(perspectiveStack, window.getChildren().get(0));
		assertEquals(1, perspectiveStack.getChildren().size());
		assertEquals(perspective, perspectiveStack.getChildren().get(0));
		assertEquals(0, perspective.getWindows().size());
	}

	public void testPerspective_Windows_ChangeWindowAttribute() {
		MApplication application = createApplication();
		MWindow window = createWindow(application);

		MPerspectiveStack perspectiveStack = AdvancedFactoryImpl.eINSTANCE
				.createPerspectiveStack();
		window.getChildren().add(perspectiveStack);

		MPerspective perspective = AdvancedFactoryImpl.eINSTANCE
				.createPerspective();
		perspectiveStack.getChildren().add(perspective);

		MWindow nestedWindow = BasicFactoryImpl.eINSTANCE.createWindow();
		nestedWindow.setElementId("nested");
		perspective.getWindows().add(nestedWindow);

		saveModel();

		ModelReconciler reconciler = createModelReconciler();
		reconciler.recordChanges(application);

		nestedWindow.setElementId("nestedWindow");

		Object state = reconciler.serialize();

		application = createApplication();
		window = application.getChildren().get(0);
		perspectiveStack = (MPerspectiveStack) window.getChildren().get(0);
		perspective = perspectiveStack.getChildren().get(0);
		nestedWindow = perspective.getWindows().get(0);

		Collection<ModelDelta> deltas = constructDeltas(application, state);

		assertEquals(1, application.getChildren().size());
		assertEquals(window, application.getChildren().get(0));
		assertEquals(1, window.getChildren().size());
		assertEquals(perspectiveStack, window.getChildren().get(0));
		assertEquals(1, perspectiveStack.getChildren().size());
		assertEquals(perspective, perspectiveStack.getChildren().get(0));
		assertEquals(1, perspective.getWindows().size());
		assertEquals(nestedWindow, perspective.getWindows().get(0));
		assertEquals("nested", nestedWindow.getElementId());

		applyAll(deltas);

		assertEquals(1, application.getChildren().size());
		assertEquals(window, application.getChildren().get(0));
		assertEquals(1, window.getChildren().size());
		assertEquals(perspectiveStack, window.getChildren().get(0));
		assertEquals(1, perspectiveStack.getChildren().size());
		assertEquals(perspective, perspectiveStack.getChildren().get(0));
		assertEquals(1, perspective.getWindows().size());
		assertEquals(nestedWindow, perspective.getWindows().get(0));
		assertEquals("nestedWindow", nestedWindow.getElementId());
	}
}
